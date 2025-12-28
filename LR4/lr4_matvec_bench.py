from mpi4py import MPI
import numpy as np
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def counts_displs(M: int, p: int):
    base, rem = divmod(M, p)
    counts = np.array([base + 1 if r < rem else base for r in range(p)], dtype=np.int64)
    displs = np.zeros(p, dtype=np.int64)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=2_000_000, help="число строк матрицы")
    parser.add_argument("--N", type=int, default=200, help="число столбцов матрицы")
    parser.add_argument("--chunk", type=int, default=50_000, help="сколько строк генерить за раз")
    parser.add_argument("--seed", type=int, default=123, help="seed для генерации A и x")
    parser.add_argument("--reps", type=int, default=1, help="сколько раз повторить вычисление (для усреднения)")
    args = parser.parse_args()

    M = int(args.M)
    N = int(args.N)
    chunk = int(args.chunk)
    reps = int(args.reps)
    seed = int(args.seed)

    counts, displs = counts_displs(M, size)
    local_M = int(counts[rank])
    start_row = int(displs[rank])

    # x генерим на root и рассылаем всем
    x = np.empty(N, dtype=np.float64)
    if rank == 0:
        rngx = np.random.default_rng(seed)
        x[:] = rngx.random(N, dtype=np.float64)
    comm.Bcast([x, N, MPI.DOUBLE], root=0)

    # вычисление b = A @ x, где A "виртуальная": генерим блоки A на лету
    comm.Barrier()
    t0 = MPI.Wtime()

    checksum_local = 0.0
    for _ in range(reps):
        done = 0
        while done < local_M:
            cur = min(chunk, local_M - done)
            global_row0 = start_row + done

            # детерминированный RNG для блока (чтобы было воспроизводимо)
            rngA = np.random.default_rng(seed + global_row0)
            A_block = rngA.random((cur, N), dtype=np.float64)

            b_block = A_block @ x
            checksum_local += float(np.sum(b_block))

            done += cur

    comm.Barrier()
    t1 = MPI.Wtime()

    # берём "время программы" как максимум по процессам (самый медленный)
    t_local = t1 - t0
    t_max = comm.allreduce(t_local, op=MPI.MAX)

    # чтобы было видно, что результат не мусор (не обязательно, но удобно)
    checksum = comm.reduce(checksum_local, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"M={M}, N={N}, procs={size}, reps={reps}")
        print(f"compute_time_max = {t_max:.6f} s")
        print(f"checksum = {checksum:.6e}")

if __name__ == "__main__":
    main()
