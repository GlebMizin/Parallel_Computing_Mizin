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


def tridiag_mv(a, b, c, x):
    """y = Ax для трёхдиагональной матрицы с диагоналями a,b,c."""
    n = x.size
    y = b * x
    y[1:] += a[1:] * x[:-1]
    y[:-1] += c[:-1] * x[1:]
    return y


def thomas(a, b, c, d):
    """
    Последовательный метод прогонки (Thomas).
    a[0]=0, c[-1]=0.
    """
    n = d.size
    cp = np.empty(n, dtype=np.float64)
    dp = np.empty(n, dtype=np.float64)

    # прямой ход
    denom = b[0]
    cp[0] = c[0] / denom if n > 1 else 0.0
    dp[0] = d[0] / denom

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    # обратный ход
    x = np.empty(n, dtype=np.float64)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20000, help="размер СЛАУ (>=10000 желательно)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    N = int(args.N)

    # ===== root генерит задачу так, чтобы решение было известно =====
    if rank == 0:
        rng = np.random.default_rng(args.seed)
        x_true = rng.standard_normal(N, dtype=np.float64)

        a = -np.ones(N, dtype=np.float64)
        b = 4.0 * np.ones(N, dtype=np.float64)
        c = -np.ones(N, dtype=np.float64)
        a[0] = 0.0
        c[-1] = 0.0

        d = tridiag_mv(a, b, c, x_true)
        counts, displs = counts_displs(N, size)
    else:
        x_true = None
        a = b = c = d = None
        counts = displs = None

    counts = comm.bcast(counts, root=0)
    displs = comm.bcast(displs, root=0)

    n_part = int(counts[rank])

    a_part = np.empty(n_part, dtype=np.float64)
    b_part = np.empty(n_part, dtype=np.float64)
    c_part = np.empty(n_part, dtype=np.float64)
    d_part = np.empty(n_part, dtype=np.float64)

    # Scatterv диагоналей и RHS
    if rank == 0:
        comm.Scatterv([a, counts, displs, MPI.DOUBLE], [a_part, n_part, MPI.DOUBLE], root=0)
        comm.Scatterv([b, counts, displs, MPI.DOUBLE], [b_part, n_part, MPI.DOUBLE], root=0)
        comm.Scatterv([c, counts, displs, MPI.DOUBLE], [c_part, n_part, MPI.DOUBLE], root=0)
        comm.Scatterv([d, counts, displs, MPI.DOUBLE], [d_part, n_part, MPI.DOUBLE], root=0)
    else:
        comm.Scatterv([None, None, None, None], [a_part, n_part, MPI.DOUBLE], root=0)
        comm.Scatterv([None, None, None, None], [b_part, n_part, MPI.DOUBLE], root=0)
        comm.Scatterv([None, None, None, None], [c_part, n_part, MPI.DOUBLE], root=0)
        comm.Scatterv([None, None, None, None], [d_part, n_part, MPI.DOUBLE], root=0)

    # Глобальные индексы начала/конца блока
    start = int(displs[rank])
    end = start + n_part - 1

    # Отдельно запоминаем "внешние" связи блока (альфа слева, гамма справа)
    alpha = float(a_part[0])   # связь с x_{start-1}
    gamma = float(c_part[-1])  # связь с x_{end+1}

    # Внутри блока эти связи убираем (как будто блока не связаны)
    a_loc = a_part.copy()
    c_loc = c_part.copy()
    a_loc[0] = 0.0
    c_loc[-1] = 0.0

    # ===== параллельная часть =====
    comm.Barrier()
    t0 = MPI.Wtime()

    # 1) y = A_k^{-1} d_k
    y = thomas(a_loc, b_part, c_loc, d_part)

    # 2) v = A_k^{-1} * [alpha,0,0,...]
    rhsL = np.zeros(n_part, dtype=np.float64)
    rhsL[0] = alpha
    v = thomas(a_loc, b_part, c_loc, rhsL)

    # 3) w = A_k^{-1} * [...,0,0,gamma]
    rhsR = np.zeros(n_part, dtype=np.float64)
    rhsR[-1] = gamma
    w = thomas(a_loc, b_part, c_loc, rhsR)

    # Коэффициенты для редуцированной системы (2*P неизвестных: u_k=x_first, v_k=x_last)
    y0, yN = float(y[0]), float(y[-1])
    v0, vN = float(v[0]), float(v[-1])
    w0, wN = float(w[0]), float(w[-1])

    coeffs = np.array([y0, yN, v0, vN, w0, wN], dtype=np.float64)

    # Собираем коэффициенты на root
    all_coeffs = None
    if rank == 0:
        all_coeffs = np.empty((size, 6), dtype=np.float64)
    comm.Gather([coeffs, 6, MPI.DOUBLE], [all_coeffs, 6, MPI.DOUBLE], root=0)

    # root решает редуцированную систему 2P x 2P и рассеивает (u_k, v_k) по процессам
    uv_local = np.empty(2, dtype=np.float64)

    if rank == 0:
        P = size
        Ared = np.zeros((2 * P, 2 * P), dtype=np.float64)
        bred = np.zeros(2 * P, dtype=np.float64)

        for k in range(P):
            y0k, yNk, v0k, vNk, w0k, wNk = all_coeffs[k]

            # Eq1: u_k + v0k * v_{k-1} + w0k * u_{k+1} = y0k
            row1 = 2 * k
            Ared[row1, 2 * k] = 1.0
            bred[row1] = y0k
            if k > 0:
                Ared[row1, 2 * (k - 1) + 1] = v0k
            if k < P - 1:
                Ared[row1, 2 * (k + 1)] = w0k

            # Eq2: v_k + vNk * v_{k-1} + wNk * u_{k+1} = yNk
            row2 = 2 * k + 1
            Ared[row2, 2 * k + 1] = 1.0
            bred[row2] = yNk
            if k > 0:
                Ared[row2, 2 * (k - 1) + 1] = vNk
            if k < P - 1:
                Ared[row2, 2 * (k + 1)] = wNk

        z = np.linalg.solve(Ared, bred)  # [u0,v0,u1,v1,...]
        uv = z.reshape(P, 2)

        comm.Scatter([uv, 2, MPI.DOUBLE], [uv_local, 2, MPI.DOUBLE], root=0)
    else:
        comm.Scatter([None, 2, MPI.DOUBLE], [uv_local, 2, MPI.DOUBLE], root=0)

    u_k = float(uv_local[0])  # x_first in block
    v_k_last = float(uv_local[1])  # x_last in block

    # ===== обмен граничными неизвестными с соседями через Sendrecv (как требуют этапы) =====
    left = rank - 1 if rank > 0 else MPI.PROC_NULL
    right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    # получить v_{k-1} (последний x предыдущего блока), отправив свой v_k вправо
    v_left = np.array(0.0, dtype=np.float64)
    send_v = np.array(v_k_last, dtype=np.float64)
    comm.Sendrecv(sendbuf=[send_v, 1, MPI.DOUBLE], dest=right, sendtag=10,
                  recvbuf=[v_left, 1, MPI.DOUBLE], source=left, recvtag=10)

    # получить u_{k+1} (первый x следующего блока), отправив свой u_k влево
    u_right = np.array(0.0, dtype=np.float64)
    send_u = np.array(u_k, dtype=np.float64)
    comm.Sendrecv(sendbuf=[send_u, 1, MPI.DOUBLE], dest=left, sendtag=20,
                  recvbuf=[u_right, 1, MPI.DOUBLE], source=right, recvtag=20)

    v_prev = float(v_left)       # v_{k-1}
    u_next = float(u_right)      # u_{k+1}

    # восстановление полного решения в блоке: x = y - v*v_prev - w*u_next
    x_part = y - v * v_prev - w * u_next

    comm.Barrier()
    t1 = MPI.Wtime()
    t_par = comm.allreduce(t1 - t0, op=MPI.MAX)

    # Собираем решение на root
    x = None
    if rank == 0:
        x = np.empty(N, dtype=np.float64)
    comm.Gatherv([x_part, n_part, MPI.DOUBLE], [x, counts, displs, MPI.DOUBLE], root=0)

    # ===== сравнение с последовательным решением на root =====
    if rank == 0:
        t2 = MPI.Wtime()
        x_seq = thomas(a, b, c, d)
        t3 = MPI.Wtime()
        t_seq = t3 - t2

        diff_par_seq = float(np.max(np.abs(x - x_seq)))
        diff_par_true = float(np.max(np.abs(x - x_true)))

        print(f"N={N}, procs={size}")
        print(f"time_parallel_max = {t_par:.6f} s")
        print(f"time_seq = {t_seq:.6f} s")
        print(f"max|x_par - x_seq|  = {diff_par_seq:.3e}")
        print(f"max|x_par - x_true| = {diff_par_true:.3e}")


if __name__ == "__main__":
    main()
