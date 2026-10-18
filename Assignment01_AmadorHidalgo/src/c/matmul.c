/*
 * matmul.c - basic dense matrix multiplication, C = A * B, n x n, float64.
 *
 * Algorithm : textbook triple loop, order i-j-k, local accumulator, O(n^3).
 * Storage   : one contiguous row-major double array per matrix (A[i*n+k]).
 * Timed part: ONLY the multiplication kernel. Inputs are read and C is
 *             allocated before any timer starts.
 *
 * Usage:
 *   matmul n A.bin B.bin warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]
 *
 *   warmup_s : keep running warm-up multiplications until this many seconds
 *              of warm-up have elapsed (0 = no warm-up; >0 = at least one run)
 *   reps     : maximum number of measured samples
 *   min_reps : samples that are always collected before the budget can stop
 *   budget_s : stop sampling once cumulative measured time exceeds this
 *   batch_s  : target duration of one sample; for very short multiplications
 *              a sample is a batch of `inner` multiplications (0 = inner 1)
 *   limit_s  : a single multiplication longer than this stops the program
 *
 * stdout (one record per line, flushed immediately):
 *   WARMUP <runs> <elapsed_ns>
 *   RESULT <rep> <inner> <batch_ns>
 *   LIMIT  <where> <ns_per_multiplication>
 *   CHECK  <sum_bits> <c00_bits> <clast_bits> <cmid_bits>   (IEEE-754 hex)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t bits(double x) {
    uint64_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}

/* The kernel under test. noinline so that repeated calls cannot be merged. */
__attribute__((noinline)) static void multiply(int n, const double *A,
                                               const double *B, double *C) {
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

static double *load(const char *path, int n) {
    size_t count = (size_t)n * (size_t)n;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    double *m = malloc(count * sizeof(double));
    if (!m) { fprintf(stderr, "out of memory\n"); exit(3); }
    if (fread(m, sizeof(double), count, f) != count) {
        fprintf(stderr, "%s: expected %zu doubles\n", path, count);
        exit(2);
    }
    fclose(f);
    return m;
}

int main(int argc, char **argv) {
    if (argc < 10) {
        fprintf(stderr, "usage: %s n A.bin B.bin warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    double warmup_s = atof(argv[4]);
    int reps = atoi(argv[5]);
    int min_reps = atoi(argv[6]);
    double budget_s = atof(argv[7]);
    double batch_s = atof(argv[8]);
    double limit_s = atof(argv[9]);
    const char *out_path = argc > 10 ? argv[10] : NULL;

    double *A = load(argv[2], n);
    double *B = load(argv[3], n);
    double *C = calloc((size_t)n * (size_t)n, sizeof(double));
    if (!C) { fprintf(stderr, "out of memory\n"); return 3; }

    /* ---- warm-up ---- */
    int warm = 0;
    uint64_t last_ns = 0, w0 = now_ns();
    while ((now_ns() - w0) * 1e-9 < warmup_s) {
        uint64_t t = now_ns();
        multiply(n, A, B, C);
        last_ns = now_ns() - t;
        warm++;
        if (last_ns * 1e-9 > limit_s) {
            printf("WARMUP %d %llu\nLIMIT warmup %llu\n", warm,
                   (unsigned long long)(now_ns() - w0), (unsigned long long)last_ns);
            fflush(stdout);
            return 0;
        }
    }
    printf("WARMUP %d %llu\n", warm, (unsigned long long)(now_ns() - w0));
    fflush(stdout);

    /* ---- batch size for very short multiplications ---- */
    long inner = 1;
    if (batch_s > 0.0) {
        if (last_ns == 0) {            /* no warm-up ran: calibrate with one run */
            uint64_t t = now_ns();
            multiply(n, A, B, C);
            last_ns = now_ns() - t;
        }
        double per = (double)last_ns * 1e-9;
        if (per > 0.0 && per < batch_s) inner = (long)(batch_s / per) + 1;
    }

    /* ---- measured samples ---- */
    double cum = 0.0;
    for (int r = 1; r <= reps; r++) {
        uint64_t t = now_ns();
        for (long b = 0; b < inner; b++) multiply(n, A, B, C);
        uint64_t dt = now_ns() - t;
        printf("RESULT %d %ld %llu\n", r, inner, (unsigned long long)dt);
        fflush(stdout);
        double per_mult = (double)dt * 1e-9 / (double)inner;
        if (per_mult > limit_s) {
            printf("LIMIT sample %llu\n", (unsigned long long)(dt / (uint64_t)inner));
            fflush(stdout);
            break;
        }
        cum += (double)dt * 1e-9;
        if (r >= min_reps && cum >= budget_s) break;
    }

    /* ---- consume the result: checksum + spot values (outside the timers) ---- */
    double sum = 0.0;
    for (size_t i = 0; i < (size_t)n * (size_t)n; i++) sum += C[i];
    double c00 = C[0];
    double clast = C[(size_t)n * n - 1];
    double cmid = C[(size_t)(n / 2) * n + (n / 3)];
    printf("CHECK %016llx %016llx %016llx %016llx\n", (unsigned long long)bits(sum),
           (unsigned long long)bits(c00), (unsigned long long)bits(clast),
           (unsigned long long)bits(cmid));
    fflush(stdout);

    if (out_path) {
        FILE *f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", out_path); return 2; }
        fwrite(C, sizeof(double), (size_t)n * (size_t)n, f);
        fclose(f);
    }
    free(A); free(B); free(C);
    return 0;
}
