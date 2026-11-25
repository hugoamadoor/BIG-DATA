/*
 * dense.c - sequential dense matrix multiplication methods, C = A * B, n x n, float64.
 *
 * Storage: one contiguous row-major double array per matrix (A[i*n+k]), as in Assignment 1.
 *
 * Methods (argument 1):
 *   ijk      Baseline. The Assignment 1 C kernel, unchanged (i-j-k, local accumulator).
 *            Inner loop walks B down a column: stride n*8 bytes.
 *   ijk_bt   Same i-j-k loops and the same summation order, but B is first transposed into a
 *            scratch buffer so that the inner loop reads both operands with stride 1.
 *            The transposition is inside the timed region.
 *   ikj      Loop reordering i-k-j: the inner loop updates a whole row of C with a scalar
 *            multiple of a row of B (C[i,:] += A[i,k] * B[k,:]). Both are stride 1 and the inner
 *            loop has no loop-carried dependence. (It could be vectorised, but GCC 13 at -O2
 *            does not do so here: build/vec_report_dense.txt lists no vectorised loop.)
 *   blocked  Cache blocking (tiling) of the i-k-j loops with square tiles of side `block`:
 *            loop order ii, kk, jj (tiles) then i, k, j (inside a tile).
 *
 * Every method adds the products for one C[i,j] in increasing k, so without fused
 * multiply-add (-ffp-contract=off) all four methods give bit-identical results.
 *
 * Timed region: the whole method call. For ikj and blocked it includes zeroing C
 * (the baseline overwrites C, so it needs no zeroing); for ijk_bt it includes the transposition.
 * Allocation of C and of the transpose buffer happens before any timer starts.
 *
 * Usage:
 *   dense method n block A.bin B.bin warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]
 * Extra record: CHECK <sum_bits> <c00_bits> <clast_bits> <cmid_bits>  (IEEE-754 hex)
 */
#include "harness.h"

typedef struct {
    int n, block;
    const double *A, *B;
    double *C, *BT;
} dense_ctx;

/* ---- baseline: Assignment 1 kernel, verbatim apart from the signature ---- */
__attribute__((noinline)) static void mm_ijk(int n, const double *restrict A,
                                             const double *restrict B, double *restrict C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            }
            C[(size_t)i * n + j] = sum;
        }
    }
}

/* ---- optimisation 1: transpose B, then i-j-k with unit stride in both operands ---- */
__attribute__((noinline)) static void mm_ijk_bt(int n, const double *restrict A,
                                                const double *restrict B, double *restrict C,
                                                double *restrict BT) {
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            BT[(size_t)j * n + k] = B[(size_t)k * n + j];
    for (int i = 0; i < n; i++) {
        const double *a = A + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double *b = BT + (size_t)j * n;
            double sum = 0.0;
            for (int k = 0; k < n; k++) sum += a[k] * b[k];
            C[(size_t)i * n + j] = sum;
        }
    }
}

/* ---- optimisation 2: loop reordering i-k-j ---- */
__attribute__((noinline)) static void mm_ikj(int n, const double *restrict A,
                                             const double *restrict B, double *restrict C) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double *c = C + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            const double a = A[(size_t)i * n + k];
            const double *b = B + (size_t)k * n;
            for (int j = 0; j < n; j++) c[j] += a * b[j];
        }
    }
}

/* ---- optimisation 3: cache blocking of i-k-j ---- */
__attribute__((noinline)) static void mm_blocked(int n, int bs, const double *restrict A,
                                                 const double *restrict B, double *restrict C) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int ii = 0; ii < n; ii += bs) {
        int ie = ii + bs < n ? ii + bs : n;
        for (int kk = 0; kk < n; kk += bs) {
            int ke = kk + bs < n ? kk + bs : n;
            for (int jj = 0; jj < n; jj += bs) {
                int je = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < ie; i++) {
                    double *c = C + (size_t)i * n;
                    for (int k = kk; k < ke; k++) {
                        const double a = A[(size_t)i * n + k];
                        const double *b = B + (size_t)k * n;
                        for (int j = jj; j < je; j++) c[j] += a * b[j];
                    }
                }
            }
        }
    }
}

static int method_id;
static void kernel(void *p) {
    dense_ctx *c = p;
    switch (method_id) {
        case 0: mm_ijk(c->n, c->A, c->B, c->C); break;
        case 1: mm_ijk_bt(c->n, c->A, c->B, c->C, c->BT); break;
        case 2: mm_ikj(c->n, c->A, c->B, c->C); break;
        case 3: mm_blocked(c->n, c->block, c->A, c->B, c->C); break;
    }
}

int main(int argc, char **argv) {
    if (argc < 12) {
        fprintf(stderr, "usage: %s ijk|ijk_bt|ikj|blocked n block A.bin B.bin "
                        "warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]\n", argv[0]);
        return 1;
    }
    const char *names[] = {"ijk", "ijk_bt", "ikj", "blocked"};
    method_id = -1;
    for (int m = 0; m < 4; m++) if (!strcmp(argv[1], names[m])) method_id = m;
    if (method_id < 0) { fprintf(stderr, "unknown method %s\n", argv[1]); return 1; }

    dense_ctx c;
    c.n = atoi(argv[2]);
    c.block = atoi(argv[3]);
    if (c.block < 1) c.block = 1;
    protocol_t p = parse_protocol(argv + 6);
    const char *out_path = argc > 12 ? argv[12] : NULL;
    size_t nn = (size_t)c.n * (size_t)c.n;

    c.A = load_dense(argv[4], c.n);
    c.B = load_dense(argv[5], c.n);
    c.C = calloc(nn, sizeof(double));
    c.BT = method_id == 1 ? calloc(nn, sizeof(double)) : NULL;
    if (!c.C || (method_id == 1 && !c.BT)) { fprintf(stderr, "out of memory\n"); return 3; }

    run_protocol(kernel, &c, p);

    /* consume the result outside the timers */
    double sum = 0.0;
    for (size_t i = 0; i < nn; i++) sum += c.C[i];
    printf("CHECK %016llx %016llx %016llx %016llx\n", (unsigned long long)bits(sum),
           (unsigned long long)bits(c.C[0]), (unsigned long long)bits(c.C[nn - 1]),
           (unsigned long long)bits(c.C[(size_t)(c.n / 2) * c.n + (c.n / 3)]));
    fflush(stdout);
    if (out_path) {
        FILE *f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", out_path); return 2; }
        fwrite(c.C, sizeof(double), nn, f);
        fclose(f);
    }
    return 0;
}
