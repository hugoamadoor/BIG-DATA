/*
 * rss_run - run a command and report the peak resident set size of that command.
 *
 * Why not read ru_maxrss from the Python harness directly? On Linux/macOS a forked child
 * inherits the parent's resident-set high-water mark until it calls exec, so a harness that
 * has NumPy loaded would add its own ~30 MB to every measurement. This wrapper is tiny
 * (about 1 MB), so the reported value is the benchmark process itself.
 *
 *   rss_run prog [args...]   ->  child's stdout/stderr unchanged, then one line
 *                                "RUSAGE_MAXRSS_KB <kilobytes>" on stdout; exit code of the child.
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
    fflush(stdout);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}
