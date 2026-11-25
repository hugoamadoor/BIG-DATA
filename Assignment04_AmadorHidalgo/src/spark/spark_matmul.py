"""Distributed matrix multiplication C = A * B on Apache Spark (RDD API), float64.

Two partitioning strategies. In both, the arithmetic inside a task is done by the blocked C kernel of
Assignments 2-3 (build/libmmkernel.so, called with ctypes), so a Spark task computes exactly what the
sequential program computes on the same data; what differs is how data and work are distributed.

rowblock  (main strategy; same decomposition as the OpenMP version of Assignment 3)
    * A is split into P row blocks (P = `parts`, block boundaries multiples of the 32-row tile).
      sc.parallelize ships one block inside each task.
    * B is sent whole to every executor with a broadcast variable (torrent broadcast: executors fetch it
      from the driver and from each other, in 4 MiB pieces).
    * Task p computes C[rows_p, :] = A[rows_p, :] * B with mm_rows and returns it.
    * The driver collects the P row blocks and stacks them into C (numpy array in driver memory).
    Communication: 8n^2 bytes of A (once), 8n^2 bytes of B per executor, 8n^2 bytes of C back.
    Result: bit-identical to the sequential baseline.

block2d  (comparison strategy: 2-D grid with a shuffle, as in MLlib's BlockMatrix.multiply)
    * A and B are cut into g x g square blocks. Block A[I,K] is replicated to the g products (I,K,J),
      block B[K,J] to the g products (I,K,J); a join brings each pair together (shuffle 1).
    * Each of the g^3 tasks computes A[I,K] * B[K,J] with mm_acc; reduceByKey((I,J)) adds the g partial
      products of each C block (shuffle 2); the driver collects the g^2 C blocks.
    Communication: 2 g * 8n^2 bytes through the join shuffle, up to g * 8n^2 through the reduce shuffle,
    8n^2 back to the driver. Result: within rounding of the sequential result (different summation order).

Every task also returns timestamps and the time spent inside the C kernel, measured inside the Python
worker, so the driver can separate kernel time from everything else (see measure()).
"""
import ctypes
import os
import time

import numpy as np

_LIB = None
BS = 32   # tile side of the C kernel (Assignment 2, experiment T)


def _lib():
    """Load the C kernel once per Python worker process."""
    global _LIB
    if _LIB is None:
        from pyspark import SparkFiles
        _LIB = ctypes.CDLL(SparkFiles.get("libmmkernel.so"))
        P = ctypes.POINTER(ctypes.c_double)
        _LIB.mm_rows.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, P, P, P]
        _LIB.mm_acc.argtypes = [ctypes.c_int] * 4 + [P, P, P]
    return _LIB


def _ptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _task_info():
    from pyspark import TaskContext
    tc = TaskContext.get()
    return dict(pid=os.getpid(), partition=tc.partitionId() if tc else -1,
                stage=tc.stageId() if tc else -1, attempt=tc.attemptNumber() if tc else -1,
                executor=os.environ.get("SPARK_EXECUTOR_ID", "?"))


# ------------------------------------------------------------------ rowblock --------------------
def _rowblock_task(item, B_bc, n):
    t0 = time.time()
    p, r0, Ablk = item
    B = B_bc.value                                  # first access in this worker fetches + unpickles B
    t1 = time.time()
    rows = Ablk.shape[0]
    C = np.empty((rows, n), dtype=np.float64)
    k0 = time.perf_counter()
    _lib().mm_rows(rows, n, BS, _ptr(Ablk), _ptr(B), _ptr(C))
    kernel_s = time.perf_counter() - k0
    t2 = time.time()
    return p, r0, C, dict(_task_info(), t_start=t0, t_got_b=t1, t_end=t2, kernel_s=kernel_s, rows=rows)


def rowblock(sc, A, B, parts):
    n = A.shape[0]
    tiles = (n + BS - 1) // BS
    bounds = [min(n, round(tiles * p / parts) * BS) for p in range(parts + 1)]
    blocks = [(p, bounds[p], np.ascontiguousarray(A[bounds[p]:bounds[p + 1]])) for p in range(parts)
              if bounds[p + 1] > bounds[p]]
    m = {}
    w0 = time.time()
    t0 = time.perf_counter()
    B_bc = sc.broadcast(B)                          # serialised and stored by the driver now
    t1 = time.perf_counter()
    w1 = time.time()
    rdd = sc.parallelize(blocks, len(blocks))
    out = rdd.map(lambda it: _rowblock_task(it, B_bc, n)).collect()   # the Spark job
    t2 = time.perf_counter()
    w2 = time.time()
    out.sort(key=lambda x: x[0])
    C = np.empty((n, n), dtype=np.float64)
    for p, r0, Cblk, _ in out:
        C[r0:r0 + Cblk.shape[0]] = Cblk
    t3 = time.perf_counter()
    B_bc.unpersist(blocking=True)
    B_bc.destroy()
    m.update(broadcast_s=t1 - t0, job_s=t2 - t1, assemble_s=t3 - t2, end_to_end_s=t3 - t0,
             wall_start=w0, job_wall_start=w1, job_wall_end=w2, tasks=[o[3] for o in out], n_tasks=len(blocks),
             bytes_A=int(A.nbytes), bytes_B_broadcast=int(B.nbytes),   # pickled size = nbytes + ~150 B header
             bytes_C=int(C.nbytes))
    return C, m


# ------------------------------------------------------------------ block2d ---------------------
def _block2d_task(kv, b):
    (I, J), (Ablk, Bblk) = kv
    t0 = time.time()
    m, q = Ablk.shape
    p = Bblk.shape[1]
    C = np.zeros((m, p), dtype=np.float64)
    k0 = time.perf_counter()
    _lib().mm_acc(m, q, p, BS, _ptr(Ablk), _ptr(Bblk), _ptr(C))
    kernel_s = time.perf_counter() - k0
    return (I, J), (C, [dict(_task_info(), t_start=t0, t_end=time.time(), kernel_s=kernel_s, rows=m)])


def _add(x, y):
    np.add(x[0], y[0], out=x[0])
    return x[0], x[1] + y[1]


def block2d(sc, A, B, g):
    n = A.shape[0]
    tiles = (n + BS - 1) // BS
    cut = [min(n, round(tiles * i / g) * BS) for i in range(g + 1)]
    blk = lambda M, I, K: np.ascontiguousarray(M[cut[I]:cut[I + 1], cut[K]:cut[K + 1]])
    a_blocks = [((I, K), blk(A, I, K)) for I in range(g) for K in range(g)]
    b_blocks = [((K, J), blk(B, K, J)) for K in range(g) for J in range(g)]
    parts = g * g
    w1 = time.time()
    t0 = time.perf_counter()
    ra = sc.parallelize(a_blocks, parts).flatMap(lambda kv: [((kv[0][0], kv[0][1], J), kv[1]) for J in range(g)])
    rb = sc.parallelize(b_blocks, parts).flatMap(lambda kv: [((I, kv[0][0], kv[0][1]), kv[1]) for I in range(g)])
    prods = ra.join(rb, g ** 3).map(lambda kv: _block2d_task(((kv[0][0], kv[0][2]), kv[1]), None))
    out = prods.reduceByKey(_add, parts).collect()
    t1 = time.perf_counter()
    w2 = time.time()
    C = np.empty((n, n), dtype=np.float64)
    tasks = []
    for (I, J), (Cblk, info) in out:
        C[cut[I]:cut[I + 1], cut[J]:cut[J + 1]] = Cblk
        tasks += info
    t2 = time.perf_counter()
    return C, dict(broadcast_s=0.0, job_s=t1 - t0, assemble_s=t2 - t1, end_to_end_s=t2 - t0, tasks=tasks,
                   wall_start=w1, job_wall_start=w1, job_wall_end=w2,
                   n_tasks=g ** 3, bytes_A=int(A.nbytes), bytes_B_broadcast=0, bytes_C=int(C.nbytes))
