/*
 * sparse.c - sparse x sparse matrix multiplication in CSR format (own code, no library).
 *
 * Format: CSR (compressed sparse row) with int64 row pointers, int32 column indices and
 * float64 values. Storage of a matrix with z nonzeros: 8(n+1) + 12z bytes.
 *
 * Algorithm: row-by-row Gustavson SpGEMM [Gustavson 1978], two passes per call:
 *   symbolic pass  - counts the distinct columns of each output row (gives row pointers),
 *   allocation     - the output arrays are allocated with the exact size,
 *   numeric pass   - C[i,:] = sum over nonzeros A[i,k] of A[i,k] * B[k,:], accumulated in a
 *                    dense accumulator of length n (a "sparse accumulator", SPA).
 * Output column indices inside a row are left in the order they were first produced (not
 * sorted). Products of the nonzeros of A[i,:] are added in increasing k, the same order as the
 * dense baseline, so for inputs without negative zeros the result is bit-identical to it.
 *
 * Timed region "multiply": free of the previous output + symbolic + allocation + numeric.
 * The SPA workspace (2n entries) is allocated once, before timing.
 * Timed region "convert" (only with dense input files): dense -> CSR for A and B, measured
 * as a separate phase so it can be reported apart from the multiplication.
 *
 * Usage:
 *   sparse dense|csr n A_file B_file warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]
 *     dense : A_file/B_file are n*n float64 row-major files; conversion is timed separately
 *     csr   : A_file/B_file are CSR files: int64 n, int64 nnz, int64 rowptr[n+1],
 *             int32 col[nnz], float64 val[nnz] (little-endian)
 *     C_out : optional, the product densified (only sensible for small n, used by the tests)
 * Records: PHASE convert|multiply, then the harness records; NNZ <A> <B> <C>; CHECK as dense.c
 */
#include "harness.h"

typedef struct {
    int64_t n, nnz;
    int64_t *rp;
    int32_t *ci;
    double *v;
} csr_t;

static void csr_free(csr_t *m) {
    free(m->rp); free(m->ci); free(m->v);
    m->rp = NULL; m->ci = NULL; m->v = NULL; m->nnz = 0;
}

static csr_t csr_from_dense(const double *D, int64_t n) {
    csr_t m;
    m.n = n;
    m.rp = xmalloc((size_t)(n + 1) * sizeof(int64_t));
    int64_t z = 0;
    m.rp[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        const double *r = D + i * n;
        for (int64_t j = 0; j < n; j++) z += (r[j] != 0.0);
        m.rp[i + 1] = z;
    }
    m.nnz = z;
    m.ci = xmalloc((size_t)z * sizeof(int32_t));
    m.v = xmalloc((size_t)z * sizeof(double));
    z = 0;
    for (int64_t i = 0; i < n; i++) {
        const double *r = D + i * n;
        for (int64_t j = 0; j < n; j++)
            if (r[j] != 0.0) { m.ci[z] = (int32_t)j; m.v[z] = r[j]; z++; }
    }
    return m;
}

static csr_t csr_load(const char *path, int64_t n_expected) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    csr_t m;
    int64_t hdr[2];
    if (fread(hdr, sizeof(int64_t), 2, f) != 2 || hdr[0] != n_expected) {
        fprintf(stderr, "%s: bad header\n", path); exit(2);
    }
    m.n = hdr[0]; m.nnz = hdr[1];
    m.rp = xmalloc((size_t)(m.n + 1) * sizeof(int64_t));
    m.ci = xmalloc((size_t)m.nnz * sizeof(int32_t));
    m.v = xmalloc((size_t)m.nnz * sizeof(double));
    if (fread(m.rp, sizeof(int64_t), (size_t)m.n + 1, f) != (size_t)m.n + 1 ||
        fread(m.ci, sizeof(int32_t), (size_t)m.nnz, f) != (size_t)m.nnz ||
        fread(m.v, sizeof(double), (size_t)m.nnz, f) != (size_t)m.nnz) {
        fprintf(stderr, "%s: truncated\n", path); exit(2);
    }
    fclose(f);
    return m;
}

typedef struct {
    int64_t n;
    const double *DA, *DB;      /* dense inputs (convert phase only) */
    csr_t A, B, C;
    double *acc;                /* SPA values  */
    int64_t *stamp;             /* SPA marker: stamp[j] == current row tag <=> j already seen */
    int32_t *cols;              /* columns produced for the current row */
    int64_t tag;
} sp_ctx;

__attribute__((noinline)) static void spgemm(sp_ctx *s) {
    const int64_t n = s->n;
    const csr_t *A = &s->A, *B = &s->B;
    csr_t *C = &s->C;
    csr_free(C);
    C->n = n;
    C->rp = xmalloc((size_t)(n + 1) * sizeof(int64_t));

    /* symbolic pass */
    int64_t z = 0;
    C->rp[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t t = ++s->tag;
        for (int64_t p = A->rp[i]; p < A->rp[i + 1]; p++) {
            int64_t k = A->ci[p];
            for (int64_t q = B->rp[k]; q < B->rp[k + 1]; q++) {
                int32_t j = B->ci[q];
                if (s->stamp[j] != t) { s->stamp[j] = t; z++; }
            }
        }
        C->rp[i + 1] = z;
    }
    C->nnz = z;
    C->ci = xmalloc((size_t)z * sizeof(int32_t));
    C->v = xmalloc((size_t)z * sizeof(double));

    /* numeric pass */
    for (int64_t i = 0; i < n; i++) {
        int64_t t = ++s->tag;
        int64_t cnt = 0;
        for (int64_t p = A->rp[i]; p < A->rp[i + 1]; p++) {
            int64_t k = A->ci[p];
            double a = A->v[p];
            for (int64_t q = B->rp[k]; q < B->rp[k + 1]; q++) {
                int32_t j = B->ci[q];
                if (s->stamp[j] != t) { s->stamp[j] = t; s->acc[j] = 0.0; s->cols[cnt++] = j; }
                s->acc[j] += a * B->v[q];
            }
        }
        int64_t base = C->rp[i];
        for (int64_t c = 0; c < cnt; c++) {
            int32_t j = s->cols[c];
            C->ci[base + c] = j;
            C->v[base + c] = s->acc[j];
        }
    }
}

static void k_multiply(void *p) { spgemm(p); }

static void k_convert(void *p) {
    sp_ctx *s = p;
    csr_free(&s->A);
    csr_free(&s->B);
    s->A = csr_from_dense(s->DA, s->n);
    s->B = csr_from_dense(s->DB, s->n);
}

static double csr_get(const csr_t *m, int64_t i, int64_t j) {
    for (int64_t p = m->rp[i]; p < m->rp[i + 1]; p++)
        if (m->ci[p] == j) return m->v[p];
    return 0.0;
}

int main(int argc, char **argv) {
    if (argc < 11) {
        fprintf(stderr, "usage: %s dense|csr n A B warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]\n", argv[0]);
        return 1;
    }
    int dense_in = !strcmp(argv[1], "dense");
    sp_ctx s;
    memset(&s, 0, sizeof s);
    s.n = atoll(argv[2]);
    protocol_t p = parse_protocol(argv + 5);
    const char *out_path = argc > 11 ? argv[11] : NULL;

    if (dense_in) {
        s.DA = load_dense(argv[3], (int)s.n);
        s.DB = load_dense(argv[4], (int)s.n);
        printf("PHASE convert\n");
        fflush(stdout);
        run_protocol(k_convert, &s, p);
    } else {
        s.A = csr_load(argv[3], s.n);
        s.B = csr_load(argv[4], s.n);
    }
    s.acc = xmalloc((size_t)s.n * sizeof(double));
    s.stamp = xmalloc((size_t)s.n * sizeof(int64_t));
    s.cols = xmalloc((size_t)s.n * sizeof(int32_t));
    for (int64_t j = 0; j < s.n; j++) s.stamp[j] = 0;

    printf("PHASE multiply\n");
    fflush(stdout);
    run_protocol(k_multiply, &s, p);

    const csr_t *C = &s.C;
    double sum = 0.0;
    for (int64_t q = 0; q < C->nnz; q++) sum += C->v[q];
    int64_t n = s.n;
    printf("NNZ %lld %lld %lld\n", (long long)s.A.nnz, (long long)s.B.nnz, (long long)C->nnz);
    printf("CHECK %016llx %016llx %016llx %016llx\n", (unsigned long long)bits(sum),
           (unsigned long long)bits(csr_get(C, 0, 0)), (unsigned long long)bits(csr_get(C, n - 1, n - 1)),
           (unsigned long long)bits(csr_get(C, n / 2, n / 3)));
    fflush(stdout);

    if (out_path) {
        double *D = calloc((size_t)n * (size_t)n, sizeof(double));
        if (!D) { fprintf(stderr, "out of memory\n"); return 3; }
        for (int64_t i = 0; i < n; i++)
            for (int64_t q = C->rp[i]; q < C->rp[i + 1]; q++) D[i * n + C->ci[q]] = C->v[q];
        FILE *f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", out_path); return 2; }
        fwrite(D, sizeof(double), (size_t)n * (size_t)n, f);
        fclose(f);
        free(D);
    }
    return 0;
}
