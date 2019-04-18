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

nx = 2**12
ny = 16
h = 1./ny

# 1D Laplacian type

diag0 = 2*np.ones(nx)
diag1 = -np.ones(nx)
Lapx = (1/h**2)*spsp.spdiags([diag1, diag0, diag1], [-1,0,1], nx,nx)
Ix = spsp.spdiags(np.ones(nx), 0, nx, nx)

diag0 = 2*np.ones(ny)
diag1 = -np.ones(ny)
Lapy = (1/h**2)*spsp.spdiags([diag1, diag0, diag1], [-1,0,1], ny,ny)
Iy = spsp.spdiags(np.ones(ny), 0, ny, ny)

Lap2D = spsp.csc_matrix(spsp.kron(Lapx, Iy) + spsp.kron(Ix, Lapy))

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