import os
import time
import argparse
import numpy as np
from mpi4py import MPI

try:
    from threadpoolctl import threadpool_info
except Exception:
    threadpool_info = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2_000_000, help="число строк матрицы (общий размер)")
    ap.add_argument("--N", type=int, default=200, help="число столбцов")
    ap.add_argument("--reps", type=int, default=5, help="число повторов ядра для замера")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    M, N = args.M, args.N
    reps = args.reps

    # делим строки между MPI процессами
    base, rem = divmod(M, size)
    local_M = base + (1 if rank < rem else 0)
    start = rank * base + min(rank, rem)

    rng = np.random.default_rng(args.seed + rank)
    A_part = rng.standard_normal((local_M, N), dtype=np.float64)
    x = np.random.default_rng(args.seed).standard_normal(N, dtype=np.float64) if rank == 0 else np.empty(N, dtype=np.float64)

    comm.Bcast(x, root=0)

    # прогрев (важно для BLAS)
    y_part = A_part @ x
    comm.Barrier()

    t0 = time.perf_counter()
    for _ in range(reps):
        y_part = A_part @ x
    comm.Barrier()
    t1 = time.perf_counter()

    t_local = t1 - t0
    t_max = comm.allreduce(t_local, op=MPI.MAX)

    checksum_local = float(np.sum(y_part))
    checksum = comm.reduce(checksum_local, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"M={M}, N={N}, procs={size}, reps={reps}")
        print(f"compute_time_max = {t_max:.6f} s")
        print(f"checksum = {checksum:.6e}")

        # покажем инфу по BLAS-потокам (если установлен threadpoolctl)
        if threadpool_info is not None:
            info = threadpool_info()
            if info:
                print("threadpools:")
                for lib in info:
                    name = lib.get("internal_api", lib.get("user_api", "lib"))
                    nthreads = lib.get("num_threads", "?")
                    impl = lib.get("filepath", "")
                    print(f"  - {name}: num_threads={nthreads} ({impl})")
            else:
                print("threadpools: <no info>")
        else:
            print("threadpools: install threadpoolctl to show BLAS threads")


if __name__ == "__main__":
    main()
