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


def u_init(x, y, eps):
    return 0.5 * np.tanh((1.0 / eps) * ((x - 0.5) ** 2 + (y - 0.5) ** 2 - 0.35 ** 2)) - 0.17

def u_left(y, t):   return 0.33
def u_right(y, t):  return 0.33
def u_bottom(x, t): return 0.33
def u_top(x, t):    return 0.33


def solve_1d(Nx, Ny, M, T=4.0, eps=10**(-1.5), mode="block"):
    a, b = -2.0, 2.0
    c, d = -2.0, 2.0

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

    # границы на t=0
    if i0 == 0:
        for j in range(1, Ny):
            u[1, j] = u_left(None, 0.0)
    if i0 + nx_loc - 1 == Nx:
        for j in range(1, Ny):
            u[nx_loc, j] = u_right(None, 0.0)

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

        # ---------- ОБМЕН GHOST ----------
        if mode == "block":
            if left != MPI.PROC_NULL:
                comm.Sendrecv(sendbuf=u[1, :], dest=left, sendtag=10,
                              recvbuf=u[0, :], source=left, recvtag=11)
            if right != MPI.PROC_NULL:
                comm.Sendrecv(sendbuf=u[nx_loc, :], dest=right, sendtag=11,
                              recvbuf=u[nx_loc + 1, :], source=right, recvtag=10)

        elif mode == "async":
            reqs = []

            # сначала Irecv (чтобы избежать дедлоков при некоторых реализациях)
            if left != MPI.PROC_NULL:
                reqs.append(comm.Irecv(u[0, :], source=left, tag=11))
            if right != MPI.PROC_NULL:
                reqs.append(comm.Irecv(u[nx_loc + 1, :], source=right, tag=10))

            # потом Isend (u мы не меняем до Wait, пишем в un)
            if left != MPI.PROC_NULL:
                reqs.append(comm.Isend(u[1, :], dest=left, tag=10))
            if right != MPI.PROC_NULL:
                reqs.append(comm.Isend(u[nx_loc, :], dest=right, tag=11))

            # пока обмен идёт — считаем “середину” (не трогаем границы блока)
            li_start = 2
            li_end = nx_loc - 1
            if li_start <= li_end:
                for li in range(li_start, li_end + 1):
                    gi = i0 + (li - 1)
                    for j in range(1, Ny):
                        if gi == 0 or gi == Nx:
                            continue
                        d2 = eps * ((u[li+1, j] - 2*u[li, j] + u[li-1, j]) / (hx*hx) +
                                    (u[li, j+1] - 2*u[li, j] + u[li, j-1]) / (hy*hy))
                        d1 = u[li, j] * ((u[li+1, j] - u[li-1, j]) / (2*hx) +
                                         (u[li, j+1] - u[li, j-1]) / (2*hy))
                        un[li, j] = u[li, j] + tau * (d2 + d1 + u[li, j]**3)

            MPI.Request.Waitall(reqs)

        else:
            raise ValueError("mode must be 'block' or 'async'")

        # ---------- ГРАНИЦЫ ----------
        # x-границы (если процесс держит глобальную границу)
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

        # ---------- ВНУТРЕННИЕ УЗЛЫ (если block — считаем всё; если async — добиваем края) ----------
        if mode == "block":
            li_start = 1
            li_end = nx_loc
        else:
            # async: середину уже посчитали, остаются li=1 и li=nx_loc (если они не глобальные границы)
            li_start = 1
            li_end = nx_loc

        for li in range(li_start, li_end + 1):
            gi = i0 + (li - 1)
            for j in range(1, Ny):
                if gi == 0 or gi == Nx:
                    continue

                # async: если это “середина”, пропускаем (уже посчитано)
                if mode == "async" and (2 <= li <= nx_loc - 1):
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

    checksum_loc = float(np.sum(u[1:nx_loc+1, :]))
    checksum = comm.reduce(checksum_loc, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"mode={mode}, Nx={Nx}, Ny={Ny}, M={M}, procs={size}")
        print(f"compute_time_max = {tmax:.6f} s")
        print(f"checksum = {checksum:.6e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["block", "async"], required=True)
    ap.add_argument("--Nx", type=int, default=50)
    ap.add_argument("--Ny", type=int, default=50)
    ap.add_argument("--M", type=int, default=500)
    args = ap.parse_args()

    solve_1d(args.Nx, args.Ny, args.M, mode=args.mode)


if __name__ == "__main__":
    main()
