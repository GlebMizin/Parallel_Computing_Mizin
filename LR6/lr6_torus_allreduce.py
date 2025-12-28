from mpi4py import MPI
import numpy as np
import argparse
import math

def torus_allreduce_sum(comm_cart: MPI.Comm, buf: np.ndarray) -> np.ndarray:
    """
    Allreduce(SUM) через 2D-тор:
    1) суммирование по строкам тора (кольцо)
    2) суммирование по столбцам тора (кольцо)
    Возвращает глобальную сумму на ВСЕХ процессах.
    """
    # подкоммуникаторы строки и столбца
    comm_row = comm_cart.Sub([False, True])   # внутри строки
    comm_col = comm_cart.Sub([True, False])   # внутри столбца

    # соседи в строке (лево/право)
    left, right = comm_row.Shift(direction=0, disp=1)
    # соседи в столбце (вверх/вниз)
    up, down = comm_col.Shift(direction=0, disp=1)

    # 1) сумма по строке
    sendbuf = buf.copy()
    sum_row = buf.copy()
    for _ in range(comm_row.Get_size() - 1):
        comm_row.Sendrecv_replace(sendbuf, dest=right, source=left)
        sum_row += sendbuf

    # 2) сумма по столбцу (по сути глобальная сумма)
    sendbuf = sum_row.copy()
    sum_all = sum_row.copy()
    for _ in range(comm_col.Get_size() - 1):
        comm_col.Sendrecv_replace(sendbuf, dest=down, source=up)
        sum_all += sendbuf

    return sum_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vec", type=int, default=1024, help="длина тестового вектора")
    parser.add_argument("--reps", type=int, default=2000, help="повторов для замера времени")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    q = int(math.isqrt(size))
    if q * q != size:
        if rank == 0:
            print("ERROR: для 2D-тора нужно число процессов = k^2 (1,4,9,16,...)")
            print(f"Сейчас size={size}")
        return

    # 2D тор
    comm_cart = comm.Create_cart(dims=(q, q), periods=(True, True), reorder=True)
    cart_rank = comm_cart.Get_rank()
    coords = comm_cart.Get_coords(cart_rank)

    # соседи в торе (по двум направлениям)
    up, down = comm_cart.Shift(direction=0, disp=1)
    left, right = comm_cart.Shift(direction=1, disp=1)

    if cart_rank == 0:
        print(f"torus dims = {q}x{q}, procs={size}")

    print(f"cart_rank={cart_rank:2d} coords={coords} "
          f"U={up:2d} D={down:2d} L={left:2d} R={right:2d}")

    # локальный тестовый вектор
    rng = np.random.default_rng(args.seed + cart_rank)
    a = rng.random(args.vec, dtype=np.float64)

    # прогрев
    comm_cart.Barrier()
    _ = torus_allreduce_sum(comm_cart, a)
    comm_cart.Barrier()
    _ = comm_cart.allreduce(a, op=MPI.SUM)
    comm_cart.Barrier()

    # замер: torus allreduce
    t0 = MPI.Wtime()
    out_torus = None
    for _ in range(args.reps):
        out_torus = torus_allreduce_sum(comm_cart, a)
    comm_cart.Barrier()
    t1 = MPI.Wtime()
    t_torus = comm.allreduce(t1 - t0, op=MPI.MAX)

    # замер: стандартный Allreduce
    t2 = MPI.Wtime()
    out_all = None
    for _ in range(args.reps):
        out_all = comm_cart.allreduce(a, op=MPI.SUM)
    comm_cart.Barrier()
    t3 = MPI.Wtime()
    t_all = comm.allreduce(t3 - t2, op=MPI.MAX)

    # проверка корректности
    diff = float(np.max(np.abs(out_torus - out_all)))

    if cart_rank == 0:
        print("----- results -----")
        print(f"vec={args.vec}, reps={args.reps}")
        print(f"time_torus_max    = {t_torus:.6f} s")
        print(f"time_allreduce_max = {t_all:.6f} s")
        print(f"max abs diff       = {diff:.3e}")

if __name__ == "__main__":
    main()
