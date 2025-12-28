import numpy as np

def main():
    # можешь поменять размеры
    M = 10  # строк матрицы
    N = 6   # столбцов матрицы
    rng = np.random.default_rng(42)

    A = rng.random((M, N), dtype=np.float64)
    x = rng.random(N, dtype=np.float64)
    b = A @ x

    # формат как в методичке: N, потом M
    with open("in.dat", "w") as f:
        f.write(f"{N}\n{M}\n")

    np.savetxt("AData.dat", A.reshape(-1), fmt="%.17e")
    np.savetxt("xData.dat", x, fmt="%.17e")
    np.savetxt("Results_seq_ref.dat", b, fmt="%.17e")
    print("Generated in.dat, AData.dat, xData.dat, Results_seq_ref.dat")

if __name__ == "__main__":
    main()
