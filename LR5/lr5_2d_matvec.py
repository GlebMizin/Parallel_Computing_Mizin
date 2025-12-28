from mpi4py import MPI
import numpy as np
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def counts_displs(n: int, p: int):
    base, rem = divmod(n, p)
    counts = np.array([base + 1 if r < rem else base for r in range(p)], dtype=np.int32)
    displs = np.zeros(p, dtype=np.int32)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=2_000_000, help="число строк A")
    parser.add_argument("--N", type=int, default=200, help="число столбцов A")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--chunk", type=int, default=50_000, help="строк A генерить за раз")
    args = parser.parse_args()

    M = int(args.M)
    N = int(args.N)
    seed = int(args.seed)
    chunk = int(args.chunk)

    # 2D сетка процессов (rows x cols)
    dims = MPI.Compute_dims(size, 2)        # подберёт близкие множители
    nrows, ncols = dims[0], dims[1]

    comm_cart = comm.Create_cart(dims=[nrows, ncols], periods=[False, False], reorder=True)
    coords = comm_cart.Get_coords(comm_cart.Get_rank())
    my_row, my_col = coords[0], coords[1]

    # Коммуникаторы строки и столбца
    comm_row = comm_cart.Sub([False, True])   # фиксируем row, меняется col
    comm_col = comm_cart.Sub([True, False])   # фиксируем col, меняется row

    # Разбиение M по строкам сетки и N по столбцам сетки
    rcounts_M, displs_M = counts_displs(M, nrows)
    rcounts_N, displs_N = counts_displs(N, ncols)

    M_part = int(rcounts_M[my_row])
    N_part = int(rcounts_N[my_col])
    row0 = int(displs_M[my_row])    # глобальная первая строка блока
    col0 = int(displs_N[my_col])    # глобальный первый столбец блока

    # x генерим на root (0,0), режем по столбцам, потом раздаём:
    # 1) Scatterv по строке my_row=0
    # 2) Bcast вниз по каждому столбцу
    x_part = np.empty(N_part, dtype=np.float64)

    if my_row == 0:
        if my_col == 0:
            rngx = np.random.default_rng(seed)
            x = rngx.random(N, dtype=np.float64)
        else:
            x = None

        # Scatterv внутри comm_row (для row=0 это отдельный коммуникатор)
        comm_row.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE],
                          [x_part, N_part, MPI.DOUBLE],
                          root=0)

    # теперь каждый столбец рассылает свой x_part вниз (root в столбце = row 0)
    comm_col.Bcast([x_part, N_part, MPI.DOUBLE], root=0)

    # ==== вычисление b = A*x при 2D разбиении ====
    # каждый процесс хранит блок A_part (M_part x N_part), считаем вклад b_temp (M_part)
    # A_part генерим на лету детерминированно (чтобы не хранить всю A)
    comm_cart.Barrier()
    t0 = MPI.Wtime()

    b_temp = np.zeros(M_part, dtype=np.float64)

    done = 0
    while done < M_part:
        cur = min(chunk, M_part - done)
        global_row0 = row0 + done

        # детерминированная генерация блока A (cur x N_part)
        # seed зависит от глобальных смещений, чтобы блоки были стабильные и разные
        rngA = np.random.default_rng(seed + global_row0 * 1_000_003 + col0 * 1_003)
        A_block = rngA.random((cur, N_part), dtype=np.float64)

        b_temp[done:done+cur] = A_block @ x_part
        done += cur

    # суммируем вклады по строке сетки (по всем столбцам): Reduce внутри comm_row
    if my_col == 0:
        b_part = np.empty(M_part, dtype=np.float64)
    else:
        b_part = None

    comm_row.Reduce([b_temp, M_part, MPI.DOUBLE],
                    [b_part, M_part, MPI.DOUBLE],
                    op=MPI.SUM, root=0)

    comm_cart.Barrier()
    t1 = MPI.Wtime()

    # время как максимум по всем процессам (самый медленный)
    t_max = comm.allreduce(t1 - t0, op=MPI.MAX)

    # собираем b на root (0,0) по столбцу col=0
    if my_col == 0:
        if my_row == 0:
            b = np.empty(M, dtype=np.float64)
        else:
            b = None

        comm_col.Gatherv([b_part, M_part, MPI.DOUBLE],
                         [b, rcounts_M, displs_M, MPI.DOUBLE],
                         root=0)
    else:
        b = None

    # печать итогов на root (0,0)
    if my_row == 0 and my_col == 0:
        checksum = float(np.sum(b))
        print(f"dims={nrows}x{ncols}, procs={size}")
        print(f"M={M}, N={N}")
        print(f"compute_time_max = {t_max:.6f} s")
        print(f"checksum = {checksum:.6e}")

if __name__ == "__main__":
    main()
