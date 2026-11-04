/*
 * harness.h - timing harness shared by dense.c and sparse.c (Assignment 2).
 *
 * It is the Assignment 1 harness (src/c/matmul.c, commit ca7622a of
 * github.com/hugoamadoor/BIG-DATA) moved into a header so that every method uses the same
 * protocol, plus one change: a hard per-multiplication time limit enforced with a timer
 * signal, so a run that exceeds the limit is stopped at the limit instead of running on.
 *
 * Records written to stdout (one per line, flushed immediately):
 *   WARMUP <runs> <elapsed_ns>
 *   RESULT <rep> <inner> <batch_ns>       one measured sample = <inner> multiplications
 *   LIMIT  <where> <ns>                   a multiplication exceeded limit_s (where = warmup|sample|killed)
 */
#ifndef HARNESS_H
#define HARNESS_H
#define _POSIX_C_SOURCE 200809L
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

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

typedef struct {
    double warmup_s, budget_s, batch_s, limit_s;
    int reps, min_reps;
} protocol_t;

static protocol_t parse_protocol(char **argv) {   /* argv: warmup reps min_reps budget batch limit */
    protocol_t p;
    p.warmup_s = atof(argv[0]);
    p.reps = atoi(argv[1]);
    p.min_reps = atoi(argv[2]);
    p.budget_s = atof(argv[3]);
    p.batch_s = atof(argv[4]);
    p.limit_s = atof(argv[5]);
    return p;
}

static char limit_msg[64];
static void on_limit(int sig) {
    (void)sig;
    ssize_t r = write(1, limit_msg, strlen(limit_msg));   /* async-signal-safe */
    (void)r;
    _exit(0);
}

static void arm(double seconds) {          /* 0 disarms */
    struct itimerval it;
    memset(&it, 0, sizeof it);
    it.it_value.tv_sec = (time_t)seconds;
    it.it_value.tv_usec = (suseconds_t)((seconds - (double)(time_t)seconds) * 1e6);
    setitimer(ITIMER_REAL, &it, NULL);
}

typedef void (*kernel_fn)(void *ctx);

/* Runs warm-up and measured samples of kernel(ctx). Returns 0 normally, 1 if a limit was hit. */
static int run_protocol(kernel_fn kernel, void *ctx, protocol_t p) {
    snprintf(limit_msg, sizeof limit_msg, "LIMIT killed %llu\n",
             (unsigned long long)(p.limit_s * 1e9));
    signal(SIGALRM, on_limit);

    /* warm-up */
    int warm = 0;
    uint64_t last_ns = 0, w0 = now_ns();
    while ((now_ns() - w0) * 1e-9 < p.warmup_s) {
        uint64_t t = now_ns();
        arm(p.limit_s);
        kernel(ctx);
        arm(0);
        last_ns = now_ns() - t;
        warm++;
    }
    printf("WARMUP %d %llu\n", warm, (unsigned long long)(now_ns() - w0));
    fflush(stdout);

    /* batch size for very short multiplications */
    long inner = 1;
    if (p.batch_s > 0.0) {
        if (last_ns == 0) {
            uint64_t t = now_ns();
            arm(p.limit_s);
            kernel(ctx);
            arm(0);
            last_ns = now_ns() - t;
        }
        double per = (double)last_ns * 1e-9;
        if (per > 0.0 && per < p.batch_s) inner = (long)(p.batch_s / per) + 1;
    }

    /* measured samples */
    double cum = 0.0;
    for (int r = 1; r <= p.reps; r++) {
        uint64_t t = now_ns();
        for (long b = 0; b < inner; b++) {
            arm(p.limit_s);
            kernel(ctx);
            arm(0);
        }
        uint64_t dt = now_ns() - t;
        printf("RESULT %d %ld %llu\n", r, inner, (unsigned long long)dt);
        fflush(stdout);
        cum += (double)dt * 1e-9;
        if (r >= p.min_reps && cum >= p.budget_s) break;
    }
    return 0;
}

static void *xmalloc(size_t bytes) {
    void *m = malloc(bytes ? bytes : 1);
    if (!m) { fprintf(stderr, "out of memory (%zu bytes)\n", bytes); exit(3); }
    return m;
}

static double *load_dense(const char *path, int n) {
    size_t count = (size_t)n * (size_t)n;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    double *m = xmalloc(count * sizeof(double));
    if (fread(m, sizeof(double), count, f) != count) {
        fprintf(stderr, "%s: expected %zu doubles\n", path, count);
        exit(2);
    }
    fclose(f);
    return m;
}

#endif
