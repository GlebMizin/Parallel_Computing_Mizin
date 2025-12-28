from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def counts_displs(M: int, p: int):
    base, rem = divmod(M, p)
    counts = np.array([base + 1 if r < rem else base for r in range(p)], dtype=np.int32)
    displs = np.zeros(p, dtype=np.int32)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs

# --- читаем размеры на root и рассылаем безопасно ---
if rank == 0:
    with open("in.dat", "r") as f:
        N = int(f.readline().strip())
        M = int(f.readline().strip())
else:
    N = None
    M = None

N = comm.bcast(N, root=0)
M = comm.bcast(M, root=0)

# counts/displs по строкам
if rank == 0:
    rcounts, rdispls = counts_displs(M, size)
else:
    rcounts = np.empty(size, dtype=np.int32)
    rdispls = np.empty(size, dtype=np.int32)

comm.Bcast([rcounts, size, MPI.INT], root=0)
comm.Bcast([rdispls, size, MPI.INT], root=0)

local_M = int(rcounts[rank])

# x везде
x = np.empty(N, dtype=np.float64)
if rank == 0:
    x[:] = np.fromfile("xData.dat", dtype=np.float64, sep=" ")
comm.Bcast([x, N, MPI.DOUBLE], root=0)

# Scatterv A (в элементах)
sendcounts_A = (rcounts * N).astype(np.int32)
displs_A = (rdispls * N).astype(np.int32)

A_part = np.empty(local_M * N, dtype=np.float64)

if rank == 0:
    A = np.fromfile("AData.dat", dtype=np.float64, sep=" ").reshape(-1)
    comm.Scatterv([A, sendcounts_A, displs_A, MPI.DOUBLE],
                  [A_part, local_M * N, MPI.DOUBLE], root=0)
else:
    comm.Scatterv([None, None, None, None],
                  [A_part, local_M * N, MPI.DOUBLE], root=0)

A_part = A_part.reshape(local_M, N)

comm.Barrier()
t0 = MPI.Wtime()
b_part = A_part @ x
comm.Barrier()
t1 = MPI.Wtime()

# Gatherv b
if rank == 0:
    b = np.empty(M, dtype=np.float64)
else:
    b = None

comm.Gatherv([b_part, local_M, MPI.DOUBLE],
             [b, rcounts, rdispls, MPI.DOUBLE], root=0)

if rank == 0:
    np.savetxt("Results_parallel.dat", b, fmt="%.17e")
    print(f"compute time ~ {t1 - t0:.6f} s, wrote Results_parallel.dat")
