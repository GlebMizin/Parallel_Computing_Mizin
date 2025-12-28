from mpi4py import MPI
import numpy as np
import argparse
from math import sin, pi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def counts_displs(n: int, p: int):
    base, rem = divmod(n, p)
    counts = np.array([base + 1 if r < rem else base for r in range(p)], dtype=np.int32)
    displs = np.zeros(p, dtype=np.int32)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs


# ==== задача как в прошлых ЛР ====
def u_init(x: float) -> float:
    return sin(3 * pi * (x - 1/6))

def u_left(t: float) -> float:
    return -1.0

def u_right(t: float) -> float:
    return +1.0


# ==== локальная (последовательная) прогонка Томаса ====
def thomas(a, b, c, d):
    """
    Решает трёхдиагональную систему:
      a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    где a[0]=0, c[-1]=0.
    """
    n = d.size
    cp = np.empty(n, dtype=np.float64)
    dp = np.empty(n, dtype=np.float64)

    denom = b[0]
    cp[0] = c[0] / denom if n > 1 else 0.0
    dp[0] = d[0] / denom

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x = np.empty(n, dtype=np.float64)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ==== параллельная прогонка (как ЛР7): решаем глобальную трёхдиагональную систему ====
def parallel_tridiag_solve(a_part, b_part, c_part, d_part, counts, displs, comm):
    """
    Решает глобальную трёхдиагональную систему, распределённую по блокам.
    a_part[0] содержит коэффициент связи с левым соседом (x_{start-1})
    c_part[-1] содержит коэффициент связи с правым соседом (x_{end+1})
    (для глобальных границ эти коэффициенты должны быть 0).
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_part = int(counts[rank])
    start = int(displs[rank])
    end = start + n_part - 1

    alpha = float(a_part[0])
    gamma = float(c_part[-1])

    a_loc = a_part.copy()
    c_loc = c_part.copy()
    a_loc[0] = 0.0
    c_loc[-1] = 0.0

    # y = A_k^{-1} d_k
    y = thomas(a_loc, b_part, c_loc, d_part)

    # v = A_k^{-1} * [alpha,0,0,...]
    rhsL = np.zeros(n_part, dtype=np.float64)
    rhsL[0] = alpha
    v = thomas(a_loc, b_part, c_loc, rhsL)

    # w = A_k^{-1} * [...,0,0,gamma]
    rhsR = np.zeros(n_part, dtype=np.float64)
    rhsR[-1] = gamma
    w = thomas(a_loc, b_part, c_loc, rhsR)

    y0, yN = float(y[0]), float(y[-1])
    v0, vN = float(v[0]), float(v[-1])
    w0, wN = float(w[0]), float(w[-1])
    coeffs = np.array([y0, yN, v0, vN, w0, wN], dtype=np.float64)

    all_coeffs = None
    if rank == 0:
        all_coeffs = np.empty((size, 6), dtype=np.float64)
    comm.Gather([coeffs, 6, MPI.DOUBLE], [all_coeffs, 6, MPI.DOUBLE], root=0)

    uv_local = np.empty(2, dtype=np.float64)
    if rank == 0:
        P = size
        Ared = np.zeros((2 * P, 2 * P), dtype=np.float64)
        bred = np.zeros(2 * P, dtype=np.float64)

        for k in range(P):
            y0k, yNk, v0k, vNk, w0k, wNk = all_coeffs[k]

            row1 = 2 * k
            Ared[row1, 2 * k] = 1.0
            bred[row1] = y0k
            if k > 0:
                Ared[row1, 2 * (k - 1) + 1] = v0k
            if k < P - 1:
                Ared[row1, 2 * (k + 1)] = w0k

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

    u_k = float(uv_local[0])       # x_first in block
    v_k_last = float(uv_local[1])  # x_last in block

    left = rank - 1 if rank > 0 else MPI.PROC_NULL
    right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    v_left = np.array(0.0, dtype=np.float64)
    send_v = np.array(v_k_last, dtype=np.float64)
    comm.Sendrecv(sendbuf=[send_v, 1, MPI.DOUBLE], dest=right, sendtag=10,
                  recvbuf=[v_left, 1, MPI.DOUBLE], source=left, recvtag=10)

    u_right = np.array(0.0, dtype=np.float64)
    send_u = np.array(u_k, dtype=np.float64)
    comm.Sendrecv(sendbuf=[send_u, 1, MPI.DOUBLE], dest=left, sendtag=20,
                  recvbuf=[u_right, 1, MPI.DOUBLE], source=right, recvtag=20)

    v_prev = float(v_left)      # v_{k-1}
    u_next = float(u_right)     # u_{k+1}

    x_part = y - v * v_prev - w * u_next
    return x_part


# ==== f(y,t) и формирование диагоналей (как в методичке для ЛР9) ====
def build_rhs_and_diags(y_loc, yL, yR, t_half, h, eps, tau, alpha, g_start, N):
    """
    y_loc: локальные значения y (без ghost), длина = local_n
    yL/yR: ghost значения соседей (скаляры) для производных на границе блока
    g_start: глобальный индекс первого элемента y_loc в векторе y (0..N-2)
    N: число интервалов по x (узлы 0..N), внутренних неизвестных N-1 => y индексы 0..N-2
    """
    local_n = y_loc.size
    a = np.empty(local_n, dtype=np.float64)
    b = np.empty(local_n, dtype=np.float64)
    c = np.empty(local_n, dtype=np.float64)
    d = np.empty(local_n, dtype=np.float64)

    # доступ к y_{g-1}, y_{g+1} для локальной границы
    def y_at(g):
        # g — глобальный индекс вектора y (0..N-2)
        if g < g_start:
            return yL
        if g >= g_start + local_n:
            return yR
        return y_loc[g - g_start]

    for i in range(local_n):
        g = g_start + i  # глобальный индекс y
        yi = y_loc[i]

        # соседние значения для производных
        if g == 0:
            y_im1 = u_left(t_half)      # вместо y[-1]
        else:
            y_im1 = y_at(g - 1)

        if g == (N - 2):
            y_ip1 = u_right(t_half)     # вместо y[N-1]
        else:
            y_ip1 = y_at(g + 1)

        # RHS f
        d2 = eps * (y_ip1 - 2.0 * yi + y_im1) / (h * h)
        d1 = yi * (y_ip1 - y_im1) / (2.0 * h)
        d[i] = d2 + d1 + yi ** 3

        # Диагонали матрицы (I - alpha*tau*f_y)
        # формулы трёхдиагональной Якоби-матрицы из методички
        if g == 0:
            a[i] = 0.0
            b[i] = 1.0 - alpha * tau * (-2.0 * eps / (h * h) + (y_ip1 - u_left(t_half)) / (2.0 * h) + 3.0 * yi ** 2)
            c[i] = -alpha * tau * (eps / (h * h) + yi / (2.0 * h))
        elif g == (N - 2):
            a[i] = -alpha * tau * (eps / (h * h) - yi / (2.0 * h))
            b[i] = 1.0 - alpha * tau * (-2.0 * eps / (h * h) + (u_right(t_half) - y_im1) / (2.0 * h) + 3.0 * yi ** 2)
            c[i] = 0.0
        else:
            a[i] = -alpha * tau * (eps / (h * h) - yi / (2.0 * h))
            b[i] = 1.0 - alpha * tau * (-2.0 * eps / (h * h) + (y_ip1 - y_im1) / (2.0 * h) + 3.0 * yi ** 2)
            c[i] = -alpha * tau * (eps / (h * h) + yi / (2.0 * h))

    return a, b, c, d


def solve_seq(N, M, a_x, b_x, t0, T, eps, alpha):
    h = (b_x - a_x) / N
    tau = (T - t0) / M
    # y = u[1..N-1], длина N-1
    y = np.empty(N - 1, dtype=np.float64)
    for g in range(N - 1):
        xg = a_x + (g + 1) * h
        y[g] = u_init(xg)

    for m in range(M):
        t_half = t0 + (m + 0.5) * tau
        # собрать a,b,c,d на весь y
        a = np.empty(N - 1, dtype=np.float64)
        b = np.empty(N - 1, dtype=np.float64)
        c = np.empty(N - 1, dtype=np.float64)
        d = np.empty(N - 1, dtype=np.float64)

        for g in range(N - 1):
            yi = y[g]
            y_im1 = u_left(t_half) if g == 0 else y[g - 1]
            y_ip1 = u_right(t_half) if g == (N - 2) else y[g + 1]

            d2 = eps * (y_ip1 - 2.0 * yi + y_im1) / (h * h)
            d1 = yi * (y_ip1 - y_im1) / (2.0 * h)
            d[g] = d2 + d1 + yi ** 3

            if g == 0:
                a[g] = 0.0
                b[g] = 1.0 - alpha * tau * (-2.0 * eps / (h * h) + (y_ip1 - u_left(t_half)) / (2.0 * h) + 3.0 * yi ** 2)
                c[g] = -alpha * tau * (eps / (h * h) + yi / (2.0 * h))
            elif g == (N - 2):
                a[g] = -alpha * tau * (eps / (h * h) - yi / (2.0 * h))
                b[g] = 1.0 - alpha * tau * (-2.0 * eps / (h * h) + (u_right(t_half) - y_im1) / (2.0 * h) + 3.0 * yi ** 2)
                c[g] = 0.0
            else:
                a[g] = -alpha * tau * (eps / (h * h) - yi / (2.0 * h))
                b[g] = 1.0 - alpha * tau * (-2.0 * eps / (h * h) + (y_ip1 - y_im1) / (2.0 * h) + 3.0 * yi ** 2)
                c[g] = -alpha * tau * (eps / (h * h) + yi / (2.0 * h))

        w1 = thomas(a, b, c, d)
        y = y + tau * w1

    return y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=20000, help="число интервалов по x (узлы 0..N)")
    p.add_argument("--M", type=int, default=5000, help="шаги по времени")
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--eps", type=float, default=10 ** (-1.5))
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--verify", action="store_true", help="root посчитает seq и сравнит")
    args = p.parse_args()

    N = int(args.N)
    M = int(args.M)
    T = float(args.T)
    eps = float(args.eps)
    alpha = float(args.alpha)

    a_x, b_x = 0.0, 1.0
    t0 = 0.0

    h = (b_x - a_x) / N
    tau = (T - t0) / M

    # неизвестные y: u(1..N-1) => длина N-1
    n_unknown = N - 1
    counts, displs = counts_displs(n_unknown, size)
    local_n = int(counts[rank])
    g_start = int(displs[rank])

    # локальный y
    y = np.empty(local_n, dtype=np.float64)
    for i in range(local_n):
        g = g_start + i
        xg = a_x + (g + 1) * h  # сдвиг на 1 (т.к. y соответствует узлам 1..N-1)
        y[i] = u_init(xg)

    # соседи
    comm_cart = comm.Create_cart(dims=[size], periods=[False], reorder=True)
    rcart = comm_cart.Get_rank()
    left = rcart - 1 if rcart > 0 else MPI.PROC_NULL
    right = rcart + 1 if rcart < size - 1 else MPI.PROC_NULL

    comm_cart.Barrier()
    t_start = MPI.Wtime()

    for m in range(M):
        t_half = t0 + (m + 0.5) * tau

        # обмен граничными y для производных (ghost)
        yL = u_left(t_half)   # если слева нет соседа и это глобальный край
        yR = u_right(t_half)  # если справа нет соседа и это глобальный край

        if left != MPI.PROC_NULL:
            send = np.array(y[0], dtype=np.float64)
            recv = np.array(0.0, dtype=np.float64)
            comm_cart.Sendrecv([send, 1, MPI.DOUBLE], dest=left, sendtag=10,
                               recvbuf=[recv, 1, MPI.DOUBLE], source=left, recvtag=11)
            yL = float(recv)

        if right != MPI.PROC_NULL:
            send = np.array(y[-1], dtype=np.float64)
            recv = np.array(0.0, dtype=np.float64)
            comm_cart.Sendrecv([send, 1, MPI.DOUBLE], dest=right, sendtag=11,
                               recvbuf=[recv, 1, MPI.DOUBLE], source=right, recvtag=10)
            yR = float(recv)

        # собрать локальные диагонали и RHS
        a_part, b_part, c_part, d_part = build_rhs_and_diags(
            y_loc=y, yL=yL, yR=yR, t_half=t_half,
            h=h, eps=eps, tau=tau, alpha=alpha,
            g_start=g_start, N=N
        )

        # решить (I - alpha*tau*J) w1 = f  параллельной прогонкой
        w1_part = parallel_tridiag_solve(a_part, b_part, c_part, d_part, counts, displs, comm)

        # обновление y_{m+1} = y_m + tau*w1
        y = y + tau * w1_part

    comm_cart.Barrier()
    t_end = MPI.Wtime()
    t_max = comm.allreduce(t_end - t_start, op=MPI.MAX)

    checksum_local = float(np.sum(y))
    checksum = comm.reduce(checksum_local, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"N={N}, M={M}, procs={size}")
        print(f"compute_time_max = {t_max:.6f} s")
        print(f"checksum = {checksum:.6e}")

    if args.verify:
        # собрать y на root и сравнить с seq
        y_all = None
        if rank == 0:
            y_all = np.empty(n_unknown, dtype=np.float64)
        comm.Gatherv([y, local_n, MPI.DOUBLE], [y_all, counts, displs, MPI.DOUBLE], root=0)

        if rank == 0:
            y_seq = solve_seq(N, M, a_x, b_x, t0, T, eps, alpha)
            diff = float(np.max(np.abs(y_all - y_seq)))
            print(f"max abs diff vs seq = {diff:.3e}")


if __name__ == "__main__":
    main()
