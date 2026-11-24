/*
 * matmul_par.c - sequential and OpenMP-parallel blocked matrix multiplication, C = A * B,
 * n x n, float64, one contiguous row-major array per matrix.
 *
 * Methods (argument 1):
 *   seq   Sequential baseline. The blocked i-k-j kernel of Assignment 2 (mm_blocked in
 *         Assignment02_AmadorHidalgo/src/dense.c, commit 1d0726d), copied unchanged. It was the
 *         fastest sequential dense method of that study.
 *   omp   The same kernel parallelised with OpenMP: the loop over row tiles (ii) is shared
 *         between threads with schedule(static). Each row tile owns a disjoint block of rows of
 *         C, which only its thread zeroes and writes, so no two threads ever write the same
 *         element and no synchronisation other than the implicit barrier at the end of the loop
 *         is needed. A and B are only read.
 *
 * Same algorithm, numeric type, storage layout, tile side and loop order inside a tile. The
 * only differences are (1) the outer tile loop is split between threads and (2) C is zeroed
 * row tile by row tile by the owning thread instead of by one memset before the loops.
 * Every C[i,j] receives its products in the same order (kk increasing, then k increasing) in
 * both methods and for every thread count, so the results are bit-identical; the tests check it.
 *
 * Thread count: OMP_NUM_THREADS (set by the harness). Thread placement: not set (OMP_PROC_BIND
 * and OMP_PLACES unset), recorded in the ENV record.
 * Timed region: the whole method call (zeroing + loops). The OpenMP thread pool is created by
 * the first parallel region, which happens in the untimed warm-up or calibration call.
 *
 * Usage:
 *   matmul_par seq|omp|tiles n block A.bin B.bin warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]
 * Records: ENV ..., the harness records (WARMUP, RESULT, LIMIT), KCPU <wall_s> <user_s> <sys_s> of the
 * whole kernel phase, THREADS <threads that executed the last parallel region>, then CHECK <sum> <c00> <clast> <cmid> (IEEE-754 hex).
 */
#include "harness.h"
#include <sys/resource.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    int n, block;
    const double *A, *B;
    double *C;
} ctx_t;

/* ---- sequential baseline: Assignment 2 mm_blocked, verbatim ---- */
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

static int threads_used = 1;

/* One row tile [ii, ie) of the blocked kernel: zero its rows of C, then the kk, jj, i, k, j loops
   of mm_blocked unchanged. A separate function with restrict parameters, because GCC drops the
   restrict qualifier of variables captured by an OpenMP region and would otherwise add a run-time
   aliasing check before the vectorised inner loop. */
__attribute__((noinline)) static void row_tile(int n, int bs, int ii, int ie, const double *restrict A,
                                               const double *restrict B, double *restrict C) {
    memset(C + (size_t)ii * n, 0, (size_t)(ie - ii) * n * sizeof(double));
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

/* ---- parallel version: row tiles shared between threads ---- */
__attribute__((noinline)) static void mm_blocked_omp(int n, int bs, const double *restrict A,
                                                     const double *restrict B, double *restrict C) {
    const int tiles = (n + bs - 1) / bs;
#pragma omp parallel
    {
#ifdef _OPENMP
#pragma omp single nowait
        threads_used = omp_get_num_threads();
#endif
#pragma omp for schedule(static)
        for (int t = 0; t < tiles; t++) {
            const int ii = t * bs;
            row_tile(n, bs, ii, ii + bs < n ? ii + bs : n, A, B, C);
        }
    }
}

/* ---- diagnostic: the parallel code path without OpenMP (row_tile called in a plain loop).
   Separates the cost of the code restructuring from the cost of the OpenMP runtime. ---- */
__attribute__((noinline)) static void mm_blocked_tiles(int n, int bs, const double *restrict A,
                                                       const double *restrict B, double *restrict C) {
    for (int ii = 0; ii < n; ii += bs) row_tile(n, bs, ii, ii + bs < n ? ii + bs : n, A, B, C);
}

static int method;   /* 0 seq, 1 omp, 2 tiles */
static void kernel(void *p) {
    ctx_t *c = p;
    if (method == 1) mm_blocked_omp(c->n, c->block, c->A, c->B, c->C);
    else if (method == 2) mm_blocked_tiles(c->n, c->block, c->A, c->B, c->C);
    else mm_blocked(c->n, c->block, c->A, c->B, c->C);
}

int main(int argc, char **argv) {
    if (argc < 12) {
        fprintf(stderr, "usage: %s seq|omp|tiles n block A.bin B.bin warmup_s reps min_reps budget_s "
                        "batch_s limit_s [C_out.bin]\n", argv[0]);
        return 1;
    }
    if (!strcmp(argv[1], "omp")) method = 1;
    else if (!strcmp(argv[1], "tiles")) method = 2;
    else if (strcmp(argv[1], "seq")) { fprintf(stderr, "unknown method %s\n", argv[1]); return 1; }

    ctx_t c;
    c.n = atoi(argv[2]);
    c.block = atoi(argv[3]);
    if (c.block < 1) c.block = 1;
    protocol_t p = parse_protocol(argv + 6);
    const char *out_path = argc > 12 ? argv[12] : NULL;
    size_t nn = (size_t)c.n * (size_t)c.n;

#ifdef _OPENMP
    printf("ENV openmp=%d max_threads=%d dynamic=%d proc_bind=%d procs=%d\n", _OPENMP,
           omp_get_max_threads(), omp_get_dynamic(), (int)omp_get_proc_bind(), omp_get_num_procs());
#else
    printf("ENV openmp=0\n");
#endif
    c.A = load_dense(argv[4], c.n);
    c.B = load_dense(argv[5], c.n);
    c.C = calloc(nn, sizeof(double));
    if (!c.C) { fprintf(stderr, "out of memory\n"); return 3; }

    /* CPU time of all threads over the kernel phase (warm-up + calibration + samples) */
    struct rusage r0, r1;
    getrusage(RUSAGE_SELF, &r0);
    uint64_t w0 = now_ns();
    run_protocol(kernel, &c, p);
    uint64_t w1 = now_ns();
    getrusage(RUSAGE_SELF, &r1);
    printf("KCPU %.6f %.6f %.6f\n", (double)(w1 - w0) * 1e-9,
           (r1.ru_utime.tv_sec - r0.ru_utime.tv_sec) + (r1.ru_utime.tv_usec - r0.ru_utime.tv_usec) * 1e-6,
           (r1.ru_stime.tv_sec - r0.ru_stime.tv_sec) + (r1.ru_stime.tv_usec - r0.ru_stime.tv_usec) * 1e-6);
    printf("THREADS %d\n", method == 1 ? threads_used : 1);

    double sum = 0.0;                      /* consume the result outside the timers */
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
