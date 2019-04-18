import numpy as np
import numpy.linalg as la
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
import time

import sys
# adding src to the path
sys.path.insert(0, '../src')

# importing the Cyclic matrix module 
from CyclicMatrix import *

nx = 2**10
ny = 2**4
h = 1./ny
# We define the system (in this case a simple 2D heat equation)


def laplacian_1d(n: int, h: float):
    # function to compute the 1d Laplacian
    diag0 = 2*np.ones(n)
    diag1 = -np.ones(n)
    Lap = (1/h**2)*spsp.spdiags([diag1, diag0, diag1], [-1,0,1], n, n)

    return Lap


def laplacian_2d(nx: int, ny: int, h: float):

    Lapx = laplacian_1d(nx, h)
    Ix = spsp.spdiags(np.ones(nx), 0, nx, nx)

    Lapy = laplacian_1d(ny, h)
    Iy = spsp.spdiags(np.ones(ny), 0, ny, ny)

    Lap2D = spsp.csc_matrix(spsp.kron(Iy, Lapx) + spsp.kron(Lapy, Ix))

    return Lap2D


Lap2D = laplacian_2d(nx, ny, h)
#
rhs = np.random.rand(Lap2D.shape[0])

# testing the construction with a sparse matrix
Csp = sp2cyclic_sparse(Lap2D, ny, nx)
Cxsp = Csp.dot(rhs)

print(la.norm(Cxsp.reshape(rhs.shape[0]) - Lap2D.dot(rhs)))

b = Cxsp.reshape(rhs.shape[0])

# x = C.solve(b)
start = time.time()
Csp.factorize()
end = time.time()
print(end - start)

start = time.time()
xOpt = Csp.solve_opt_c(b)
print(la.norm(xOpt.reshape((-1)) - rhs))
end = time.time()
print(end - start)