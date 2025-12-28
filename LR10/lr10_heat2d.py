from mpi4py import MPI
import numpy as np
import argparse
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def counts_displs(n: int, p: int):
    base, rem = divmod(n, p)
    counts = np.array([base + 1 if r < rem else base for r in range(p)], dtype=np.int32)
    displs = np.zeros(p, dtype=np.int32)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs


# === условия как в лекционном примере для 2D задачи ===
def u_init(x, y, eps):
    return 0.5 * np.tanh((1.0 / eps) * ((x - 0.5) ** 2 + (y - 0.5) ** 2 - 0.35 ** 2)) - 0.17

def u_left(y, t):   return 0.33
def u_right(y, t):  return 0.33
def u_bottom(x, t): return 0.33
def u_top(x, t):    return 0.33


def step_explicit(u, un, hx, hy, tau, eps, tnext):
    # u, un: (Nx+1, Ny+1)
    Nx = u.shape[0] - 1
    Ny = u.shape[1] - 1

    # границы
    for j in range(1, Ny):
        un[0, j] = u_left(None, tnext)
        un[Nx, j] = u_right(None, tnext)
    for i in range(0, Nx + 1):
        un[i, 0] = u_bottom(None, tnext)
        un[i, Ny] = u_top(None, tnext)

    # внутренние
    for i in range(1, Nx):
        for j in range(1, Ny):
            d2 = eps * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (hx*hx) +
                        (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / (hy*hy))
            d1 = u[i, j] * ((u[i+1, j] - u[i-1, j]) / (2*hx) +
                            (u[i, j+1] - u[i, j-1]) / (2*hy))
            un[i, j] = u[i, j] + tau * (d2 + d1 + u[i, j]**3)


def solve_seq(Nx, Ny, M, a, b, c, d, T, eps):
    x = np.linspace(a, b, Nx + 1)
    y = np.linspace(c, d, Ny + 1)
    tau = T / M
    hx = (b - a) / Nx
    hy = (d - c) / Ny

    u = np.empty((Nx + 1, Ny + 1), dtype=np.float64)
    un = np.empty_like(u)

    # init
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            u[i, j] = u_init(x[i], y[j], eps)

    # границы на t=0
    for j in range(1, Ny):
        u[0, j] = u_left(y[j], 0.0)
        u[Nx, j] = u_right(y[j], 0.0)
    for i in range(0, Nx + 1):
        u[i, 0] = u_bottom(x[i], 0.0)
        u[i, Ny] = u_top(x[i], 0.0)

    for m in range(M):
        tnext = (m + 1) * tau
        step_explicit(u, un, hx, hy, tau, eps, tnext)
        u, un = un, u

    return u


# ---------- 1D декомпозиция (по x): каждый процесс хранит полосу по i ----------
def solve_1d(Nx, Ny, M, a, b, c, d, T, eps, verify=False):
    tau = T / M
    hx = (b - a) / Nx
    hy = (d - c) / Ny

    # делим i=0..Nx (Nx+1 узлов) между процессами
    total_i = Nx + 1
    cnt_i, dsp_i = counts_displs(total_i, size)
    nx_loc = int(cnt_i[rank])
    i0 = int(dsp_i[rank])  # глобальный i начала

    # локальный массив: (nx_loc + 2) x (Ny+1), ghost-строки по i
    u = np.empty((nx_loc + 2, Ny + 1), dtype=np.float64)
    un = np.empty_like(u)

    # init
    for li in range(1, nx_loc + 1):
        gi = i0 + (li - 1)
        xg = a + gi * hx
        for j in range(Ny + 1):
            yg = c + j * hy
            u[li, j] = u_init(xg, yg, eps)

    # границы на t=0 (если попали в глобальные края)
    if i0 == 0:
        for j in range(1, Ny):
            u[1, j] = u_left(None, 0.0)
    if i0 + nx_loc - 1 == Nx:
        for j in range(1, Ny):
            u[nx_loc, j] = u_right(None, 0.0)
    # низ/верх всегда локально
    for li in range(1, nx_loc + 1):
        gi = i0 + (li - 1)
        xg = a + gi * hx
        u[li, 0] = u_bottom(xg, 0.0)
        u[li, Ny] = u_top(xg, 0.0)

    left = rank - 1 if rank > 0 else MPI.PROC_NULL
    right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    comm.Barrier()
    t0 = MPI.Wtime()

    for m in range(M):
        tnext = (m + 1) * tau

        # обмен ghost-строками: отправляем первую и последнюю реальные строки
        # чтобы сосед мог заполнить ghost
        if left != MPI.PROC_NULL:
            comm.Sendrecv(sendbuf=u[1, :], dest=left, sendtag=10,
                          recvbuf=u[0, :], source=left, recvtag=11)
        if right != MPI.PROC_NULL:
            comm.Sendrecv(sendbuf=u[nx_loc, :], dest=right, sendtag=11,
                          recvbuf=u[nx_loc + 1, :], source=right, recvtag=10)

        # считаем следующий слой
        # границы по x (если глобальные)
        if i0 == 0:
            for j in range(1, Ny):
                un[1, j] = u_left(None, tnext)
        if i0 + nx_loc - 1 == Nx:
            for j in range(1, Ny):
                un[nx_loc, j] = u_right(None, tnext)

        # низ/верх
        for li in range(1, nx_loc + 1):
            gi = i0 + (li - 1)
            xg = a + gi * hx
            un[li, 0] = u_bottom(xg, tnext)
            un[li, Ny] = u_top(xg, tnext)

        # внутренние узлы
        for li in range(1, nx_loc + 1):
            gi = i0 + (li - 1)
            for j in range(1, Ny):
                # если глобальная левая/правая граница — уже обработали
                if gi == 0 or gi == Nx:
                    continue
                d2 = eps * ((u[li+1, j] - 2*u[li, j] + u[li-1, j]) / (hx*hx) +
                            (u[li, j+1] - 2*u[li, j] + u[li, j-1]) / (hy*hy))
                d1 = u[li, j] * ((u[li+1, j] - u[li-1, j]) / (2*hx) +
                                 (u[li, j+1] - u[li, j-1]) / (2*hy))
                un[li, j] = u[li, j] + tau * (d2 + d1 + u[li, j]**3)

        u, un = un, u

    comm.Barrier()
    t1 = MPI.Wtime()
    tmax = comm.allreduce(t1 - t0, op=MPI.MAX)

    # checksum
    checksum_loc = float(np.sum(u[1:nx_loc+1, :]))
    checksum = comm.reduce(checksum_loc, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"mode=1d, Nx={Nx}, Ny={Ny}, M={M}, procs={size}")
        print(f"compute_time_max = {tmax:.6f} s")
        print(f"checksum = {checksum:.6e}")

    # verify: собрать и сравнить с seq (только для маленьких сеток!)
    if verify:
        # собрать u на root (по полосам)
        if rank == 0:
            u_all = np.empty((Nx + 1, Ny + 1), dtype=np.float64)
        else:
            u_all = None

        # собираем по i (полосы), отправляем без ghost
        send = u[1:nx_loc+1, :].copy()

        # gatherv по строкам: сложнее из-за 2D, делаем ручной Gather на root
        parts = comm.gather(send, root=0)
        if rank == 0:
            # склеим по i
            cur = 0
            for p, block in enumerate(parts):
                u_all[cur:cur + block.shape[0], :] = block
                cur += block.shape[0]
            u_seq = solve_seq(Nx, Ny, M, a, b, c, d, T, eps)
            diff = float(np.max(np.abs(u_all - u_seq)))
            print(f"max abs diff vs seq = {diff:.3e}")


# ---------- 2D декомпозиция: q x q, только если procs = квадрат ----------
def solve_2d(Nx, Ny, M, a, b, c, d, T, eps):
    q = int(math.isqrt(size))
    if q * q != size:
        if rank == 0:
            print("ERROR: mode=2d требует число процессов = k^2 (1,4,9,16,...)")
            print(f"Сейчас procs={size}")
        return

    tau = T / M
    hx = (b - a) / Nx
    hy = (d - c) / Ny

    # 2D сетка процессов
    comm_cart = comm.Create_cart(dims=(q, q), periods=(False, False), reorder=True)
    r = comm_cart.Get_rank()
    coords = comm_cart.Get_coords(r)
    pr, pc = coords[0], coords[1]

    # делим i=0..Nx на q блоков, j=0..Ny на q блоков
    cnt_i, dsp_i = counts_displs(Nx + 1, q)
    cnt_j, dsp_j = counts_displs(Ny + 1, q)
    nx_loc = int(cnt_i[pr]); i0 = int(dsp_i[pr])
    ny_loc = int(cnt_j[pc]); j0 = int(dsp_j[pc])

    # локальный блок + halo: (nx_loc+2) x (ny_loc+2)
    u = np.empty((nx_loc + 2, ny_loc + 2), dtype=np.float64)
    un = np.empty_like(u)

    # init
    for li in range(1, nx_loc + 1):
        gi = i0 + (li - 1)
        xg = a + gi * hx
        for lj in range(1, ny_loc + 1):
            gj = j0 + (lj - 1)
            yg = c + gj * hy
            u[li, lj] = u_init(xg, yg, eps)

    # граничные условия на t=0 (если блок касается глобальной границы)
    if i0 == 0:
        for lj in range(1, ny_loc + 1):
            u[1, lj] = u_left(None, 0.0)
    if i0 + nx_loc - 1 == Nx:
        for lj in range(1, ny_loc + 1):
            u[nx_loc, lj] = u_right(None, 0.0)
    if j0 == 0:
        for li in range(1, nx_loc + 1):
            gi = i0 + (li - 1)
            xg = a + gi * hx
            u[li, 1] = u_bottom(xg, 0.0)
    if j0 + ny_loc - 1 == Ny:
        for li in range(1, nx_loc + 1):
            gi = i0 + (li - 1)
            xg = a + gi * hx
            u[li, ny_loc] = u_top(xg, 0.0)

    up, down = comm_cart.Shift(0, 1)
    left, right = comm_cart.Shift(1, 1)

    comm_cart.Barrier()
    t0 = MPI.Wtime()

    for m in range(M):
        tnext = (m + 1) * tau

        # обмен halo: строки вверх/вниз
        if up != MPI.PROC_NULL:
            comm_cart.Sendrecv(sendbuf=u[1, 1:ny_loc+1], dest=up, sendtag=10,
                               recvbuf=u[0, 1:ny_loc+1], source=up, recvtag=11)
        if down != MPI.PROC_NULL:
            comm_cart.Sendrecv(sendbuf=u[nx_loc, 1:ny_loc+1], dest=down, sendtag=11,
                               recvbuf=u[nx_loc+1, 1:ny_loc+1], source=down, recvtag=10)

        # обмен halo: столбцы влево/вправо (нужно паковать в contiguous буфер)
        if left != MPI.PROC_NULL:
            send_col = np.ascontiguousarray(u[1:nx_loc + 1, 1])
            recv_col = np.empty(nx_loc, dtype=np.float64)
            comm_cart.Sendrecv(sendbuf=send_col, dest=left, sendtag=20,
                               recvbuf=recv_col, source=left, recvtag=21)
            u[1:nx_loc + 1, 0] = recv_col

        if right != MPI.PROC_NULL:
            send_col = np.ascontiguousarray(u[1:nx_loc + 1, ny_loc])
            recv_col = np.empty(nx_loc, dtype=np.float64)
            comm_cart.Sendrecv(sendbuf=send_col, dest=right, sendtag=21,
                               recvbuf=recv_col, source=right, recvtag=20)
            u[1:nx_loc + 1, ny_loc + 1] = recv_col

        # границы на tnext (если есть)
        if i0 == 0:
            for lj in range(1, ny_loc + 1):
                un[1, lj] = u_left(None, tnext)
        if i0 + nx_loc - 1 == Nx:
            for lj in range(1, ny_loc + 1):
                un[nx_loc, lj] = u_right(None, tnext)
        if j0 == 0:
            for li in range(1, nx_loc + 1):
                gi = i0 + (li - 1)
                xg = a + gi * hx
                un[li, 1] = u_bottom(xg, tnext)
        if j0 + ny_loc - 1 == Ny:
            for li in range(1, nx_loc + 1):
                gi = i0 + (li - 1)
                xg = a + gi * hx
                un[li, ny_loc] = u_top(xg, tnext)

        # внутренние узлы локального блока
        for li in range(1, nx_loc + 1):
            gi = i0 + (li - 1)
            for lj in range(1, ny_loc + 1):
                gj = j0 + (lj - 1)
                if gi == 0 or gi == Nx or gj == 0 or gj == Ny:
                    continue
                d2 = eps * ((u[li+1, lj] - 2*u[li, lj] + u[li-1, lj]) / (hx*hx) +
                            (u[li, lj+1] - 2*u[li, lj] + u[li, lj-1]) / (hy*hy))
                d1 = u[li, lj] * ((u[li+1, lj] - u[li-1, lj]) / (2*hx) +
                                 (u[li, lj+1] - u[li, lj-1]) / (2*hy))
                un[li, lj] = u[li, lj] + tau * (d2 + d1 + u[li, lj]**3)

        u, un = un, u

    comm_cart.Barrier()
    t1 = MPI.Wtime()
    tmax = comm.allreduce(t1 - t0, op=MPI.MAX)

    checksum_loc = float(np.sum(u[1:nx_loc+1, 1:ny_loc+1]))
    checksum = comm.reduce(checksum_loc, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"mode=2d, grid={q}x{q}, Nx={Nx}, Ny={Ny}, M={M}, procs={size}")
        print(f"compute_time_max = {tmax:.6f} s")
        print(f"checksum = {checksum:.6e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seq", "1d", "2d"], required=True)
    ap.add_argument("--Nx", type=int, default=50)
    ap.add_argument("--Ny", type=int, default=50)
    ap.add_argument("--M", type=int, default=500)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--eps", type=float, default=10 ** (-1.5))
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    a, b = -2.0, 2.0
    c, d = -2.0, 2.0

    if args.mode == "seq":
        if rank == 0:
            t0 = MPI.Wtime()
            u = solve_seq(args.Nx, args.Ny, args.M, a, b, c, d, args.T, args.eps)
            t1 = MPI.Wtime()
            print(f"mode=seq, Nx={args.Nx}, Ny={args.Ny}, M={args.M}")
            print(f"time = {t1 - t0:.6f} s")
            print(f"checksum = {float(np.sum(u)):.6e}")
        return

    if args.mode == "1d":
        solve_1d(args.Nx, args.Ny, args.M, a, b, c, d, args.T, args.eps, verify=args.verify)
        return

    if args.mode == "2d":
        solve_2d(args.Nx, args.Ny, args.M, a, b, c, d, args.T, args.eps)
        return


if __name__ == "__main__":
    main()
