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

# Граничные и начальное условия (как в примере из методички)
def u_init(x: float) -> float:
    return sin(3 * pi * (x - 1/6))

def u_left(t: float) -> float:
    return -1.0

def u_right(t: float) -> float:
    return +1.0

def step_explicit(u_prev, u_next, h, tau, eps, start_idx, N, t_next):
    """
    u_prev: локальный массив с ghost-ячейками (len = local_n + 2)
    u_next: локальный массив с ghost-ячейками
    start_idx: глобальный индекс первого локального узла
    N: число интервалов (узлы 0..N)
    """
    local_n = u_prev.size - 2
    # Считаем узлы i=1..local_n (соответствуют глобальным start_idx..start_idx+local_n-1)
    for i in range(1, local_n + 1):
        g = start_idx + (i - 1)
        if g == 0:
            u_next[i] = u_left(t_next)
            continue
        if g == N:
            u_next[i] = u_right(t_next)
            continue

        d2 = (u_prev[i + 1] - 2.0 * u_prev[i] + u_prev[i - 1]) / (h * h)
        d1 = (u_prev[i + 1] - u_prev[i - 1]) / (2.0 * h)
        u_next[i] = (
            u_prev[i]
            + eps * tau * d2
            + tau * u_prev[i] * d1
            + tau * (u_prev[i] ** 3)
        )

def solve_seq(N, M, eps, a, b, t0, T):
    x = np.linspace(a, b, N + 1)
    t = np.linspace(t0, T, M + 1)
    h = (b - a) / N
    tau = (T - t0) / M

    u_prev = np.empty(N + 1, dtype=np.float64)
    u_next = np.empty(N + 1, dtype=np.float64)

    # init
    for n in range(N + 1):
        u_prev[n] = u_init(x[n])
    u_prev[0] = u_left(t[0])
    u_prev[-1] = u_right(t[0])

    for m in range(M):
        tn1 = t[m + 1]
        u_next[0] = u_left(tn1)
        u_next[-1] = u_right(tn1)
        for n in range(1, N):
            d2 = (u_prev[n + 1] - 2.0 * u_prev[n] + u_prev[n - 1]) / (h * h)
            d1 = (u_prev[n + 1] - u_prev[n - 1]) / (2.0 * h)
            u_next[n] = u_prev[n] + eps * tau * d2 + tau * u_prev[n] * d1 + tau * (u_prev[n] ** 3)
        u_prev, u_next = u_next, u_prev

    return u_prev

def main():
    p = argparse.ArgumentParser()
    # В методичке для больших тестов N=800, M=300000 (очень тяжело на ПК) :contentReference[oaicite:1]{index=1}
    p.add_argument("--N", type=int, default=800, help="число интервалов по x (узлов N+1)")
    p.add_argument("--M", type=int, default=20000, help="число шагов по времени")
    p.add_argument("--eps", type=float, default=10 ** (-1.5))
    p.add_argument("--a", type=float, default=0.0)
    p.add_argument("--b", type=float, default=1.0)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--T", type=float, default=6.0)
    p.add_argument("--verify", action="store_true", help="root дополнительно посчитает seq и сравнит")
    args = p.parse_args()

    N = int(args.N)
    M = int(args.M)
    eps = float(args.eps)
    a = float(args.a)
    b = float(args.b)
    t0 = float(args.t0)
    T = float(args.T)

    # сетка
    h = (b - a) / N
    tau = (T - t0) / M

    # делим узлы 0..N (всего N+1 узел) между процессами
    total_nodes = N + 1
    counts, displs = counts_displs(total_nodes, size)
    local_n = int(counts[rank])
    start = int(displs[rank])  # глобальный индекс первого узла в блоке

    # локальные массивы + 2 ghost-ячейки
    u_prev = np.zeros(local_n + 2, dtype=np.float64)
    u_next = np.zeros(local_n + 2, dtype=np.float64)

    # инициализация u(x, t0)
    # hookup: локальный узел i соответствует global g = start + (i-1)
    for i in range(1, local_n + 1):
        g = start + (i - 1)
        xg = a + g * h
        u_prev[i] = u_init(xg)

    # граничные условия на t0 (если у процесса есть узлы 0 или N)
    if start == 0:
        u_prev[1] = u_left(t0)
    if start + local_n - 1 == N:
        u_prev[local_n] = u_right(t0)

    # топология-линейка
    comm_cart = comm.Create_cart(dims=[size], periods=[False], reorder=True)
    rcart = comm_cart.Get_rank()

    left = rcart - 1 if rcart > 0 else MPI.PROC_NULL
    right = rcart + 1 if rcart < size - 1 else MPI.PROC_NULL

    comm_cart.Barrier()
    t_start = MPI.Wtime()

    # основной цикл
    for m in range(M):
        t_next = t0 + (m + 1) * tau

        # 1) обмен граничными значениями текущего слоя (u_prev) для заполнения ghost
        # левый ghost (u_prev[0]) <- последний узел левого соседа
        if left != MPI.PROC_NULL:
            send_left = np.array(u_prev[1], dtype=np.float64)
            recv_left = np.array(0.0, dtype=np.float64)
            comm_cart.Sendrecv(sendbuf=[send_left, 1, MPI.DOUBLE], dest=left, sendtag=10,
                               recvbuf=[recv_left, 1, MPI.DOUBLE], source=left, recvtag=11)
            u_prev[0] = float(recv_left)

        # правый ghost (u_prev[-1]) <- первый узел правого соседа
        if right != MPI.PROC_NULL:
            send_right = np.array(u_prev[local_n], dtype=np.float64)
            recv_right = np.array(0.0, dtype=np.float64)
            comm_cart.Sendrecv(sendbuf=[send_right, 1, MPI.DOUBLE], dest=right, sendtag=11,
                               recvbuf=[recv_right, 1, MPI.DOUBLE], source=right, recvtag=10)
            u_prev[local_n + 1] = float(recv_right)

        # 2) граничные условия на слое m+1 (только для узлов 0 и N)
        if start == 0:
            u_next[1] = u_left(t_next)
        if start + local_n - 1 == N:
            u_next[local_n] = u_right(t_next)

        # 3) расчёт следующего слоя на своём блоке
        step_explicit(u_prev, u_next, h, tau, eps, start, N, t_next)

        # swap
        u_prev, u_next = u_next, u_prev

    comm_cart.Barrier()
    t_end = MPI.Wtime()
    t_local = t_end - t_start
    t_max = comm.allreduce(t_local, op=MPI.MAX)

    # checksum финального слоя
    checksum_local = float(np.sum(u_prev[1:local_n + 1]))
    checksum = comm.reduce(checksum_local, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"N={N}, M={M}, procs={size}")
        print(f"compute_time_max = {t_max:.6f} s")
        print(f"checksum = {checksum:.6e}")

    # опциональная проверка: root считает последовательную версию и сравнивает
    if args.verify:
        # собрать u(T) на root
        u_T = None
        if rank == 0:
            u_T = np.empty(total_nodes, dtype=np.float64)
        comm.Gatherv([u_prev[1:local_n + 1], local_n, MPI.DOUBLE],
                     [u_T, counts, displs, MPI.DOUBLE], root=0)

        if rank == 0:
            u_seq = solve_seq(N, M, eps, a, b, t0, T)
            diff = float(np.max(np.abs(u_T - u_seq)))
            print(f"max abs diff vs seq = {diff:.3e}")

if __name__ == "__main__":
    main()
