import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * MatMul - basic dense matrix multiplication, C = A * B, n x n, float64.
 *
 * Algorithm : textbook triple loop, order i-j-k, local accumulator, O(n^3).
 * Storage   : double[n][n] (array of row arrays; each row is contiguous).
 * Timed part: ONLY the multiplication kernel. Inputs are read and C is
 *             allocated before any timer starts.
 *
 * Usage:
 *   java MatMul n A.bin B.bin warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]
 * Arguments and stdout records are identical to src/c/matmul.c.
 */
public final class MatMul {

    /** The kernel under test. */
    static void multiply(int n, double[][] A, double[][] B, double[][] C) {
        for (int i = 0; i < n; i++) {
            double[] Ai = A[i];
            double[] Ci = C[i];
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += Ai[k] * B[k][j];
                }
                Ci[j] = sum;
            }
        }
    }

    static double[][] load(String path, int n) throws IOException {
        double[][] m = new double[n][n];
        ByteBuffer row = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
        try (FileChannel ch = FileChannel.open(Path.of(path), StandardOpenOption.READ)) {
            if (ch.size() != 8L * n * n) {
                throw new IOException(path + ": expected " + (8L * n * n) + " bytes, found " + ch.size());
            }
            for (int i = 0; i < n; i++) {
                row.clear();
                while (row.hasRemaining()) {
                    if (ch.read(row) < 0) throw new IOException("unexpected end of file");
                }
                row.flip();
                row.asDoubleBuffer().get(m[i]);
            }
        }
        return m;
    }

    static String hex(double x) {
        return String.format("%016x", Double.doubleToLongBits(x));
    }

    public static void main(String[] a) throws IOException {
        if (a.length < 9) {
            System.err.println("usage: java MatMul n A.bin B.bin warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]");
            System.exit(1);
        }
        int n = Integer.parseInt(a[0]);
        double warmupS = Double.parseDouble(a[3]);
        int reps = Integer.parseInt(a[4]);
        int minReps = Integer.parseInt(a[5]);
        double budgetS = Double.parseDouble(a[6]);
        double batchS = Double.parseDouble(a[7]);
        double limitS = Double.parseDouble(a[8]);
        String outPath = a.length > 9 ? a[9] : null;

        double[][] A = load(a[1], n);
        double[][] B = load(a[2], n);
        double[][] C = new double[n][n];

        /* ---- warm-up (lets the JIT compile the kernel) ---- */
        int warm = 0;
        long lastNs = 0, w0 = System.nanoTime();
        while ((System.nanoTime() - w0) * 1e-9 < warmupS) {
            long t = System.nanoTime();
            multiply(n, A, B, C);
            lastNs = System.nanoTime() - t;
            warm++;
            if (lastNs * 1e-9 > limitS) {
                System.out.println("WARMUP " + warm + " " + (System.nanoTime() - w0));
                System.out.println("LIMIT warmup " + lastNs);
                System.out.flush();
                return;
            }
        }
        System.out.println("WARMUP " + warm + " " + (System.nanoTime() - w0));
        System.out.flush();

        /* ---- batch size for very short multiplications ---- */
        long inner = 1;
        if (batchS > 0.0) {
            if (lastNs == 0) {
                long t = System.nanoTime();
                multiply(n, A, B, C);
                lastNs = System.nanoTime() - t;
            }
            double per = lastNs * 1e-9;
            if (per > 0.0 && per < batchS) inner = (long) (batchS / per) + 1;
        }

        /* ---- measured samples ---- */
        double cum = 0.0;
        for (int r = 1; r <= reps; r++) {
            long t = System.nanoTime();
            for (long b = 0; b < inner; b++) multiply(n, A, B, C);
            long dt = System.nanoTime() - t;
            System.out.println("RESULT " + r + " " + inner + " " + dt);
            System.out.flush();
            double perMult = dt * 1e-9 / inner;
            if (perMult > limitS) {
                System.out.println("LIMIT sample " + (dt / inner));
                System.out.flush();
                break;
            }
            cum += dt * 1e-9;
            if (r >= minReps && cum >= budgetS) break;
        }

        /* ---- consume the result: checksum + spot values (outside the timers) ---- */
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double[] Ci = C[i];
            for (int j = 0; j < n; j++) sum += Ci[j];
        }
        System.out.println("CHECK " + hex(sum) + " " + hex(C[0][0]) + " "
                + hex(C[n - 1][n - 1]) + " " + hex(C[n / 2][n / 3]));
        System.out.flush();

        if (outPath != null) {
            ByteBuffer row = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
            try (FileChannel ch = FileChannel.open(Path.of(outPath), StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
                for (int i = 0; i < n; i++) {
                    row.clear();
                    row.asDoubleBuffer().put(C[i]);
                    row.limit(8 * n);
                    row.position(0);
                    while (row.hasRemaining()) ch.write(row);
                }
            }
        }
    }
}
