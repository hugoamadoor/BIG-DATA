"""matmul.py - basic dense matrix multiplication, C = A * B, n x n, float64.

Algorithm : textbook triple loop, order i-j-k, local accumulator, O(n^3),
            written in pure Python (no NumPy, no library multiplication).
Storage   : list of lists of Python floats (CPython's native 2-D structure;
            every element is a separate boxed float object).
Timed part: ONLY the multiplication kernel. Inputs are read and C is
            allocated before any timer starts.

Usage:
  python3 matmul.py n A.bin B.bin warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]
Arguments and stdout records are identical to src/c/matmul.c.
"""
import array
import struct
import sys
import time


def multiply(n, A, B, C):
    """The kernel under test."""
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += Ai[k] * B[k][j]
            Ci[j] = s


def load(path, n):
    flat = array.array("d")
    with open(path, "rb") as f:
        flat.fromfile(f, n * n)          # raises EOFError if the file is short
    if sys.byteorder != "little":
        flat.byteswap()
    return [flat[i * n:(i + 1) * n].tolist() for i in range(n)]


def bits(x):
    return "%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


def main(argv):
    if len(argv) < 10:
        sys.stderr.write("usage: matmul.py n A.bin B.bin warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]\n")
        return 1
    n = int(argv[1])
    warmup_s = float(argv[4])
    reps = int(argv[5])
    min_reps = int(argv[6])
    budget_s = float(argv[7])
    batch_s = float(argv[8])
    limit_s = float(argv[9])
    out_path = argv[10] if len(argv) > 10 else None
    clock = time.perf_counter_ns         # monotonic

    A = load(argv[2], n)
    B = load(argv[3], n)
    C = [[0.0] * n for _ in range(n)]

    # ---- warm-up ----
    warm = 0
    last_ns = 0
    w0 = clock()
    while (clock() - w0) * 1e-9 < warmup_s:
        t = clock()
        multiply(n, A, B, C)
        last_ns = clock() - t
        warm += 1
        if last_ns * 1e-9 > limit_s:
            print("WARMUP %d %d" % (warm, clock() - w0))
            print("LIMIT warmup %d" % last_ns)
            sys.stdout.flush()
            return 0
    print("WARMUP %d %d" % (warm, clock() - w0))
    sys.stdout.flush()

    # ---- batch size for very short multiplications ----
    inner = 1
    if batch_s > 0.0:
        if last_ns == 0:
            t = clock()
            multiply(n, A, B, C)
            last_ns = clock() - t
        per = last_ns * 1e-9
        if 0.0 < per < batch_s:
            inner = int(batch_s / per) + 1

    # ---- measured samples ----
    cum = 0.0
    for r in range(1, reps + 1):
        t = clock()
        for _ in range(inner):
            multiply(n, A, B, C)
        dt = clock() - t
        print("RESULT %d %d %d" % (r, inner, dt))
        sys.stdout.flush()
        if dt * 1e-9 / inner > limit_s:
            print("LIMIT sample %d" % (dt // inner))
            sys.stdout.flush()
            break
        cum += dt * 1e-9
        if r >= min_reps and cum >= budget_s:
            break

    # ---- consume the result: checksum + spot values (outside the timers) ----
    total = 0.0
    for row in C:
        for v in row:
            total += v
    print("CHECK %s %s %s %s" % (bits(total), bits(C[0][0]), bits(C[n - 1][n - 1]),
                                  bits(C[n // 2][n // 3])))
    sys.stdout.flush()

    if out_path:
        out = array.array("d")
        for row in C:
            out.extend(row)
        if sys.byteorder != "little":
            out.byteswap()
        with open(out_path, "wb") as f:
            out.tofile(f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
