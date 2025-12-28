import numpy as np

with open("in.dat", "r") as f:
    N = int(f.readline().strip())
    M = int(f.readline().strip())

A = np.fromfile("AData.dat", dtype=np.float64, sep=" ").reshape(M, N)
x = np.fromfile("xData.dat", dtype=np.float64, sep=" ")

b = A @ x
np.savetxt("Results_seq.dat", b, fmt="%.17e")
print("Wrote Results_seq.dat")
