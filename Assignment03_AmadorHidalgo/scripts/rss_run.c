/*
 * rss_run - run a command and report the peak resident set size of that command.
 *
 * Why not read ru_maxrss from the Python harness directly? On Linux/macOS a forked child
 * inherits the parent's resident-set high-water mark until it calls exec, so a harness that
 * has NumPy loaded would add its own ~30 MB to every measurement. This wrapper is tiny
 * (about 1 MB), so the reported value is the benchmark process itself.
 *
 *   rss_run prog [args...]   ->  child's stdout/stderr unchanged, then one line
 *                                "RUSAGE_MAXRSS_KB <kilobytes>" and "RUSAGE_CPU_S <user> <sys>"
 *                                on stdout; exit code of the child.
 * Assignment 3 adds the CPU-time line (user and system seconds of all threads of the child),
 * used to estimate how many cores a process kept busy on average.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: rss_run prog [args...]\n"); return 2; }
    pid_t p = fork();
    if (p == 0) {
        execvp(argv[1], argv + 1);
        perror("exec");
        _exit(127);
    }
    int st = 0;
    struct rusage ru;
    wait4(p, &st, 0, &ru);
    long kb = ru.ru_maxrss;
#ifdef __APPLE__
    kb /= 1024;          /* macOS reports bytes, Linux kilobytes */
#endif
    printf("RUSAGE_MAXRSS_KB %ld\n", kb);
    printf("RUSAGE_CPU_S %.6f %.6f\n", ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6,
           ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6);
    fflush(stdout);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}
