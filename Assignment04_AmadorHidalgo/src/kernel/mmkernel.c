/*
 * mmkernel.c - the block kernel executed inside every Spark task (shared library, called with ctypes).
 *
 * It is the blocked i-k-j kernel of Assignments 2 and 3 (32 x 32 tiles; row_tile in
 * Assignment03_AmadorHidalgo/src/matmul_par.c), compiled with the same flags, so that a Spark task
 * does exactly the arithmetic the sequential and OpenMP versions do on the same rows.
 *
 *   mm_rows : C[rows x n] = A[rows x n] * B[n x n]          (row-block strategy)
 *             Row tiles of bs rows, then kk, jj, i, k, j as in row_tile. Each C[i,j] receives its
 *             products in increasing k, exactly as in the sequential baseline, so the assembled C is
 *             bit-identical to it whatever the number of partitions (as long as partition boundaries
 *             are multiples of bs, which the driver guarantees).
 *   mm_acc  : C[m x p] += A[m x q] * B[q x p]               (2-D block strategy)
 *             Same loop nest on rectangular blocks, accumulating into C. The partial products of
 *             different k-blocks are added later by Spark (reduceByKey) in an order Spark chooses,
 *             so this strategy is checked against a tolerance, not bit for bit.
 *
 * All arrays are contiguous, row-major float64. Build: see build.sh (-O2 -ffp-contract=off -fPIC).
 */
#include <stddef.h>
#include <string.h>

static inline int min_i(int a, int b) { return a < b ? a : b; }

void mm_rows(int rows, int n, int bs, const double *restrict A, const double *restrict B,
             double *restrict C) {
    memset(C, 0, (size_t)rows * n * sizeof(double));
    for (int ii = 0; ii < rows; ii += bs) {
        int ie = min_i(ii + bs, rows);
        for (int kk = 0; kk < n; kk += bs) {
            int ke = min_i(kk + bs, n);
            for (int jj = 0; jj < n; jj += bs) {
                int je = min_i(jj + bs, n);
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

void mm_acc(int m, int q, int p, int bs, const double *restrict A, const double *restrict B,
            double *restrict C) {
    for (int ii = 0; ii < m; ii += bs) {
        int ie = min_i(ii + bs, m);
        for (int kk = 0; kk < q; kk += bs) {
            int ke = min_i(kk + bs, q);
            for (int jj = 0; jj < p; jj += bs) {
                int je = min_i(jj + bs, p);
                for (int i = ii; i < ie; i++) {
                    double *c = C + (size_t)i * p;
                    for (int k = kk; k < ke; k++) {
                        const double a = A[(size_t)i * q + k];
                        const double *b = B + (size_t)k * p;
                        for (int j = jj; j < je; j++) c[j] += a * b[j];
                    }
                }
            }
        }
    }
}
