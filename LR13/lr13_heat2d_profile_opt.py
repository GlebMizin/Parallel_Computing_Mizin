import argparse
import os
import cProfile, pstats
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def counts_displs(n: int, p: int):
    base, rem = divmod(n, p)
    counts = np.array([base + 1 if r < rem else base for r in range(p)], dtype=np.int32)
    displs = np.zeros(p, dtype=np.int32)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs


# те же функции/константы что в ЛР10/11
def u_init(x, y, eps):
    return 0.5 * np.tanh((1.0 / eps) * ((x - 0.5) ** 2 + (y - 0.5) ** 2 - 0.35 ** 2)) - 0.17

def u_left(t):   return 0.33
def u_right(t):  return 0.33
def u_bottom(t): return 0.33
def u_top(t):    return 0.33


def step_py(u, un, nx_loc, Ny, hx, hy, tau, eps, i0, Nx, tnext):
    # границы по x (если процесс держит глобальную)
    if i0 == 0:
        un[1, 1:Ny] = u_left(tnext)
    if i0 + nx_loc - 1 == Nx:
        un[nx_loc, 1:Ny] = u_right(tnext)

    # низ/верх
    un[1:nx_loc+1, 0] = u_bottom(tnext)
    un[1:nx_loc+1, Ny] = u_top(tnext)

    # внутренние узлы
    for li in range(1, nx_loc + 1):
        gi = i0 + (li - 1)
        if gi == 0 or gi == Nx:
            continue
        for j in range(1, Ny):
            d2 = eps * ((u[li+1, j] - 2*u[li, j] + u[li-1, j]) / (hx*hx) +
                        (u[li, j+1] - 2*u[li, j] + u[li, j-1]) / (hy*hy))
            d1 = u[li, j] * ((u[li+1, j] - u[li-1, j]) / (2*hx) +
                             (u[li, j+1] - u[li, j-1]) / (2*hy))
            un[li, j] = u[li, j] + tau * (d2 + d1 + u[li, j]**3)


def step_vec(u, un, nx_loc, Ny, hx, hy, tau, eps, i0, Nx, tnext):
    # границы по x (если процесс держит глобальную)
    if i0 == 0:
        un[1, 1:Ny] = u_left(tnext)
    if i0 + nx_loc - 1 == Nx:
        un[nx_loc, 1:Ny] = u_right(tnext)

    # низ/верх
    un[1:nx_loc+1, 0] = u_bottom(tnext)
    un[1:nx_loc+1, Ny] = u_top(tnext)

    # какие local i реально считаем (чтобы не трогать глобальные границы)
    li_start = 1
    li_end = nx_loc
    if i0 == 0:
        li_start = 2
    if i0 + nx_loc - 1 == Nx:
        li_end = nx_loc - 1

    if li_start > li_end:
        return  # нечего считать

    # срезы для внутренней области
    U = u[li_start:li_end+1, 1:Ny]
    Uxp = u[li_start+1:li_end+2, 1:Ny]
    Uxm = u[li_start-1:li_end,   1:Ny]
    Uyp = u[li_start:li_end+1, 2:Ny+1]
    Uym = u[li_start:li_end+1, 0:Ny-1]

    d2 = eps * ((Uxp - 2*U + Uxm) / (hx*hx) + (Uyp - 2*U + Uym) / (hy*hy))
    d1 = U * ((Uxp - Uxm) / (2*hx) + (Uyp - Uym) / (2*hy))
    un[li_start:li_end+1, 1:Ny] = U + tau * (d2 + d1 + U**3)


def solve(Nx, Ny, M, kernel="py", comm_mode="block"):
    # область как в ЛР10
    a, b = -2.0, 2.0
    c, d = -2.0, 2.0
    T = 4.0
    eps = 10 ** (-1.5)

    tau = T / M
    hx = (b - a) / Nx
    hy = (d - c) / Ny

    # делим i=0..Nx (Nx+1 узлов) между процессами
    total_i = Nx + 1
    cnt_i, dsp_i = counts_displs(total_i, size)
    nx_loc = int(cnt_i[rank])
    i0 = int(dsp_i[rank])

    # локальные массивы + ghost
    u = np.empty((nx_loc + 2, Ny + 1), dtype=np.float64)
    un = np.empty_like(u)

    # init
    for li in range(1, nx_loc + 1):
        gi = i0 + (li - 1)
        xg = a + gi * hx
        for j in range(Ny + 1):
            yg = c + j * hy
            u[li, j] = u_init(xg, yg, eps)

    # границы t=0
    if i0 == 0:
        u[1, 1:Ny] = u_left(0.0)
    if i0 + nx_loc - 1 == Nx:
        u[nx_loc, 1:Ny] = u_right(0.0)
    u[1:nx_loc+1, 0] = u_bottom(0.0)
    u[1:nx_loc+1, Ny] = u_top(0.0)

    left = rank - 1 if rank > 0 else MPI.PROC_NULL
    right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    step_fn = step_py if kernel == "py" else step_vec

    comm.Barrier()
    t0 = MPI.Wtime()

    for m in range(M):
        tnext = (m + 1) * tau

        # обмен ghost строками
        if comm_mode == "block":
            if left != MPI.PROC_NULL:
                comm.Sendrecv(sendbuf=u[1, :], dest=left, sendtag=10,
                              recvbuf=u[0, :], source=left, recvtag=11)
            if right != MPI.PROC_NULL:
                comm.Sendrecv(sendbuf=u[nx_loc, :], dest=right, sendtag=11,
                              recvbuf=u[nx_loc + 1, :], source=right, recvtag=10)

        elif comm_mode == "async":
            reqs = []
            if left != MPI.PROC_NULL:
                reqs.append(comm.Irecv(u[0, :], source=left, tag=11))
            if right != MPI.PROC_NULL:
                reqs.append(comm.Irecv(u[nx_loc + 1, :], source=right, tag=10))
            if left != MPI.PROC_NULL:
                reqs.append(comm.Isend(u[1, :], dest=left, tag=10))
            if right != MPI.PROC_NULL:
                reqs.append(comm.Isend(u[nx_loc, :], dest=right, tag=11))
            MPI.Request.Waitall(reqs)
        else:
            raise ValueError("comm_mode must be block or async")

        # шаг схемы
        step_fn(u, un, nx_loc, Ny, hx, hy, tau, eps, i0, Nx, tnext)
        u, un = un, u

    comm.Barrier()
    t1 = MPI.Wtime()

    t_local = t1 - t0
    t_max = comm.allreduce(t_local, op=MPI.MAX)

    checksum_local = float(np.sum(u[1:nx_loc+1, :]))
    checksum = comm.reduce(checksum_local, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"kernel={kernel}, comm={comm_mode}, Nx={Nx}, Ny={Ny}, M={M}, procs={size}")
        print(f"compute_time_max = {t_max:.6f} s")
        print(f"checksum = {checksum:.6e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Nx", type=int, default=50)
    ap.add_argument("--Ny", type=int, default=50)
    ap.add_argument("--M", type=int, default=500)
    ap.add_argument("--kernel", choices=["py", "vec"], required=True)
    ap.add_argument("--comm", choices=["block", "async"], default="block")
    ap.add_argument("--profile", action="store_true", help="cProfile на rank0 (и только на нём)")
    args = ap.parse_args()

    if args.profile and rank == 0:
        prof = cProfile.Profile()
        prof.enable()
        solve(args.Nx, args.Ny, args.M, kernel=args.kernel, comm_mode=args.comm)
        prof.disable()
        out = f"profile_rank0_{args.kernel}_{args.comm}_p{size}.pstats"
        prof.dump_stats(out)
        # короткий топ-10 в txt
        with open(out.replace(".pstats", ".txt"), "w", encoding="utf-8") as f:
            ps = pstats.Stats(prof, stream=f).sort_stats("cumtime")
            ps.print_stats(15)
        print(f"wrote {out} and {out.replace('.pstats','.txt')}")
    else:
        solve(args.Nx, args.Ny, args.M, kernel=args.kernel, comm_mode=args.comm)


if __name__ == "__main__":
    main()
