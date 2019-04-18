import numpy as np
import numpy.linalg as la
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
from numba import jit, prange
from numba import jitclass

from numba import config, njit, threading_layer
config.THREADING_LAYER = 'threadsafe'

@njit
def idxBlk(i: int, n: int) -> np.array:
    return i * n + np.arange(n)


def idxBLK(i, j, n):
    return np.ix_(idxBlk(i, n), idxBlk(j, n))


def M2Cyclic(M, n, nblocks):
    # M is supposed to be tridiagonal
    assert M.shape[0] == n * nblocks
    assert M.shape[1] == n * nblocks

    E, D, F = [], [], []

    for ii in range(nblocks):
        # we introduced two dummy matrices for E and F
        D.append(M[idxBLK(ii, ii, n)])
        if ii > 0:
            E.append(M[idxBLK(ii, ii - 1, n)])
        else:  # dummy matrix
            # E.append(spsp.csr_matrix((n, n), dtype=float))
            E.append(np.zeros((n, n)))
        if ii < nblocks - 1:
            F.append(M[idxBLK(ii, ii + 1, n)])
        else:  # dummy matrix
            F.append(np.zeros((n, n)))  # this allows to use numba

    return CyclicMatrix(n, nblocks, E, D, F)


def sp2cyclic_sparse(M, n, nblocks):
    # M is supposed to be tridiagonal
    assert M.shape[0] == n * nblocks
    assert M.shape[1] == n * nblocks

    E, D, F = [], [], []

    for ii in range(nblocks):
        # we introduced two dummy matrices for E and F
        D.append(spsp.csc_matrix(M[idxBLK(ii, ii, n)]))
        if ii > 0:
            E.append(spsp.csc_matrix(M[idxBLK(ii, ii - 1, n)]))
        else:
            E.append(spsp.csc_matrix((n, n), dtype=float))
        if ii < nblocks - 1:
            F.append(spsp.csc_matrix(M[idxBLK(ii, ii + 1, n)]))
        else:
            F.append(spsp.csc_matrix((n, n), dtype=float))

    return CyclicSparseMatrix(n, nblocks, E, D, F)


class CyclicMatrix:
    def __init__(self, n, nblocks, E, D, F):
        # we want to make sure that the size of the blocks are good
        assert nblocks == len(E) and nblocks == len(D) and nblocks == len(F)
        self.n = n  # size of each block
        self.nblocks = nblocks
        self.E = E
        self.D = D
        self.F = F
        self.Dinv = []
        self.C = []

    def dot_slow(self, X):
        # dot product with a vector
        assert (X.shape[0] == self.n * self.nblocks)
        # we reshape it in order to be sure that it will be a 2D array
        X = np.reshape(X, (X.shape[0], -1))
        # we allocate the answer
        Ax = np.zeros(X.shape)

        for ii in range(self.nblocks):
            Ax[idxBlk(ii, self.n), :] += self.D[ii].dot(X[idxBlk(ii, self.n), :])
            if ii > 0:
                Ax[idxBlk(ii, self.n), :] += self.E[ii].dot(X[idxBlk(ii - 1, self.n), :])
            if ii < self.nblocks - 1:
                Ax[idxBlk(ii, self.n), :] += self.F[ii].dot(X[idxBlk(ii + 1, self.n), :])

        return Ax

    # this one is definitely faster
    def dot(self, X):
        # dot product with a vector
        assert (X.shape[0] == self.n * self.nblocks)
        # we reshape it in order to be sure that it will be a 2D array
        X = np.reshape(X, (X.shape[0], -1))
        # we allocate the answer
        Ax = np.zeros(X.shape)

        Ax[idxBlk(0, self.n), :] = self.D[0] @ X[idxBlk(0, self.n), :] + \
                                   self.F[0] @ X[idxBlk(0 + 1, self.n), :]

        for ii in range(1, self.nblocks-1):
            Ax[idxBlk(ii, self.n), :] = self.D[ii] @ X[idxBlk(ii, self.n), :] + \
                                        self.E[ii] @ X[idxBlk(ii - 1, self.n), :] +\
                                        self.F[ii] @ X[idxBlk(ii + 1, self.n), :]

        Ax[idxBlk(self.nblocks-1, self.n), :] = self.D[self.nblocks-1] @ X[idxBlk(self.nblocks-1, self.n), :] + \
                                       self.E[self.nblocks-1] @ X[idxBlk(self.nblocks-2, self.n), :]

        return Ax

    def factorize(self):
        # function to solve Cx = b, using cyclic reduction
        if len(self.Dinv) != 0:
            print("the matrix is already factorized")
        elif self.nblocks == 1:
            # if we are in a leaf, i.e. we are dealing with a dense matrix
            self.Dinv.append(la.inv(self.D[0]))
        else:
            # defing the indices
            IdxO = np.arange(0, self.nblocks, 2)
            IdxE = np.arange(1, self.nblocks, 2)

            # we start inverting the diagonal blocks
            Dinv = []

            for ii in range(len(IdxO)):
                Dinv.append(la.inv(self.D[IdxO[ii]]))

            E = []
            D = []
            F = []

            for ii in range(len(IdxE)):

                jj = IdxE[ii]
                Dlocal = np.zeros(self.D[jj].shape)

                Dlocal += self.D[jj]

                if jj > 0:  # this should never be false, but just in case
                    # update the diagonal
                    Dlocal -= self.E[jj].dot(Dinv[ii].dot(self.F[jj - 1]))
                if jj < self.nblocks - 1:  # this can be true
                    # update the diagonal
                    Dlocal -= self.F[jj].dot(Dinv[ii + 1].dot(self.E[jj + 1]))

                # now treating the off-diagonal blocks
                if jj > 0 and ii > 0:
                    Elocal = - self.E[jj].dot(Dinv[ii].dot(self.E[jj - 1]))
                else:
                    Elocal = np.zeros((self.n, self.n))

                if jj < self.nblocks - 1 and ii < len(IdxE) - 1:
                    Flocal = - self.F[jj].dot(Dinv[ii + 1].dot(self.F[jj + 1]))
                else:
                    Flocal = np.zeros((self.n, self.n))

                E.append(Elocal)
                D.append(Dlocal)
                F.append(Flocal)

            # defining the local inverses
            self.Dinv = Dinv
            # building the matrix for the resulting problem
            self.C = CyclicMatrix(self.n, len(IdxE), E, D, F)
            #factorizing the new matrix
            self.C.factorize()

    def solve_opt_c(self, b):
        assert self.n * self.nblocks == b.shape[0]
        b = np.reshape(b, (self.n * self.nblocks, -1))
        m = b.shape[1]

        if len(self.Dinv) == 0:
            print("Matrix was not factorized, we proceed to factorize it")
            self.factorize()

        # this is going to be called recursively
        if self.nblocks == 1:
            return self.Dinv[0].dot(b)

        # function to solve Cx = b, using cyclic reduction
        IdxO = np.arange(0, self.nblocks, 2)
        IdxE = np.arange(1, self.nblocks, 2)

        # defining the local solutions
        xO = np.zeros((self.n * len(IdxO), m))
        # defining the right hand side for the
        # Schur complement
        bE = np.zeros((self.n * len(IdxE), m))
        # allocating the final answer
        x = np.zeros((self.n * self.nblocks, m))

        for ii in range(len(IdxO)):
            xO[idxBlk(ii, self.n), :] = self.Dinv[ii].dot(b[idxBlk(IdxO[ii], self.n), :])

        for ii in range(len(IdxE)):
            jj = IdxE[ii]
            bE[idxBlk(ii, self.n), :] = b[idxBlk(jj, self.n), :]

            if jj > 0:  # this should never be false, but just in case
                # update the rhs
                bE[idxBlk(ii, self.n), :] -= self.E[jj].dot(xO[idxBlk(ii, self.n), :])
            if jj < self.nblocks - 1:  # this can be true
                # update the rhs
                bE[idxBlk(ii, self.n), :] -= self.F[jj].dot(xO[idxBlk(ii+1, self.n), :])

        xE = self.C.solve_opt_c(bE)

        for ii in range(len(IdxE)):
            jj = IdxE[ii]
            x[idxBlk(jj, self.n), :] = xE[idxBlk(ii, self.n), :]

        for ii in range(len(IdxO)):
            jj = IdxO[ii]
            x[idxBlk(jj, self.n), :] += xO[idxBlk(ii, self.n), :]
            if jj < self.nblocks - 1:
                x[idxBlk(jj, self.n), :] -= self.Dinv[ii].dot(self.F[jj].dot(x[idxBlk(jj+1, self.n), :]))
            if jj > 0:
                x[idxBlk(jj, self.n), :] -= self.Dinv[ii].dot(self.E[jj].dot(x[idxBlk(jj-1, self.n), :]))

        return x

    def solve_opt_numba(self, b):
        assert self.n * self.nblocks == b.shape[0]

        # if the matrix is a leaf
        if self.nblocks <= 1:
            if not self.Dinv :
                if spsp.issparse(self.D[0]):
                    self.Dinv.append(la.inv(self.D[0].toarray()))
                else:
                    self.Dinv.append(la.inv(self.D[0]))

            return self.Dinv[0].dot(b)

        # function to solve Cx = b, using cyclic reduction
        IdxO = np.arange(0, self.nblocks, 2)
        IdxE = np.arange(1, self.nblocks, 2)

        xO = np.zeros(self.n * len(IdxO))

        if not self.Dinv :
            # we start inverting the diagonal blocks
            Dinv = []

            for ii in range(len(IdxO)):
                # we check if the matrix is sparse or not
                if spsp.issparse(self.D[IdxO[ii]]):
                    Dinv.append(la.inv(self.D[IdxO[ii]].toarray()))
                else :
                    Dinv.append(la.inv(self.D[IdxO[ii]]))

            E = []
            D = []
            F = []
            for ii in range(len(IdxE)):

                jj = IdxE[ii]
                Dlocal = np.zeros(self.D[jj].shape)
                if spsp.issparse(self.D[jj]):
                    Dlocal += self.D[jj].toarray()
                else:
                    Dlocal += self.D[jj]

                if jj > 0:  # this should never be false, but just in case
                    # update the diagonal
                    if spsp.issparse(self.F[jj-1]):
                        Dlocal -= self.E[jj].toarray().dot(Dinv[ii].dot(self.F[jj-1].toarray()))
                    else :
                        Dlocal -= self.E[jj].dot(Dinv[ii].dot(self.F[jj-1]))
                if jj < self.nblocks - 1:  # this can be true
                    # update the diagonal
                    if spsp.issparse(self.E[jj+1]):
                        Dlocal -= self.F[jj].dot(Dinv[ii+1].dot( self.E[jj+1].toarray()))
                    else :
                        Dlocal -= self.F[jj].dot(Dinv[ii+1].dot( self.E[jj+1]))

                # now treating the off-diagonal blocks
                if jj > 0 and ii > 0:
                    if spsp.issparse(self.E[jj-1]):
                        Elocal = -self.E[jj].dot(Dinv[ii].dot(self.E[jj-1].toarray()))
                    else :
                        Elocal = -self.E[jj].dot(Dinv[ii].dot(self.E[jj-1]))
                else:
                    # Elocal = spsp.csr_matrix((self.n, self.n), dtype=float)
                    Elocal = np.zeros((self.n, self.n))

                if jj < self.nblocks - 1 and ii < len(IdxE) - 1:
                    if spsp.issparse(self.F[jj+1]):
                        Flocal = - self.F[jj].dot(Dinv[ii+1].dot(self.F[jj+1].toarray()))
                    else :
                        Flocal = - self.F[jj].dot(Dinv[ii+1].dot(self.F[jj+1]))
                else:
                    # Flocal = spsp.csr_matrix((self.n, self.n), dtype=float)
                    Flocal = np.zeros((self.n, self.n))

                E.append(Elocal)
                D.append(Dlocal)
                F.append(Flocal)


            # defining the reduced problem
            self.Dinv = Dinv
            self.C = CyclicMatrix(self.n, len(IdxE), E, D, F)

        ###solving

        for ii in range(len(IdxO)):
            xO[idxBlk(ii, self.n)] = self.Dinv[ii].dot(b[idxBlk(IdxO[ii], self.n)])

        bE = np.zeros(self.n * len(IdxE))

        for ii in range(len(IdxE)):

            jj = IdxE[ii]
            bE[idxBlk(ii, self.n)] = b[idxBlk(jj, self.n)]

            if jj > 0:  # this should never be false, but just in case
                # update the rhs
                bE[idxBlk(ii, self.n)] -= self.E[jj].dot(xO[idxBlk(ii, self.n)])
            if jj < self.nblocks - 1:  # this can be true
                # update the rhs
                bE[idxBlk(ii, self.n)] -= self.F[jj].dot(xO[idxBlk(ii+1, self.n)])

        xE = self.C.solve_opt_numba(bE)

        x = np.zeros(self.n * self.nblocks)

        # for ii in range(len(IdxE)):
        #     jj = IdxE[ii]
        #     x[idxBlk(jj, self.n)] = xE[idxBlk(ii, self.n)]
        # for ii in range(len(IdxO)):
        #     jj = IdxO[ii]
        #     x[idxBlk(jj, self.n)] = xO[idxBlk(ii, self.n)]

        transcribe_numba(x, xE, IdxE, self.n)
        transcribe_numba(x, xO, IdxO, self.n)

        for ii in range(len(IdxO)):
            jj = IdxO[ii]
            if jj < self.nblocks - 1:
                x[idxBlk(jj, self.n)] -= self.Dinv[ii].dot(self.F[jj].dot(x[idxBlk(jj+1, self.n)]))
            if jj > 0:
                x[idxBlk(jj, self.n)] -= self.Dinv[ii].dot(self.E[jj].dot(x[idxBlk(jj-1, self.n)]))



        # backsubs(x, xE, self.E, self.Dinv, self.F, IdxO, self.n, self.nblocks)

        return x


# def transcribe(x, xE, IdxE, n):
#     for ii in range(len(IdxE)):
#         jj = IdxE[ii]
#         x[idxBlk(jj, n)] = xE[idxBlk(ii, n)]

@jit(nopython=True, parallel=True)
def transcribe_numba(x, xE, IdxE, n):
    for ii in range(len(IdxE)):
        jj = IdxE[ii]
        x[idxBlk(jj, n)] = xE[idxBlk(ii, n)]

def solve(xO, IdxO, Dinv, n):
    for ii in range(len(IdxO)):
        xO[idxBlk(ii, n)] = Dinv[ii].dot(b[idxBlk(IdxO[ii], n)])

@jit(nopython=True, parallel=True)
def solve_numba(xO, IdxO, Dinv, n):
    for ii in range(len(IdxO)):
        xO[idxBlk(ii, n)] = Dinv[ii,:,:].dot(b[idxBlk(IdxO[ii], n)])

@jit(nopython=True)
def backsubs(x, xO, E, Dinv, F, IdxO, n, nblocks):
    for ii in range(len(IdxO)):
        jj = IdxO[ii]
        if jj < nblocks - 1:
            x[idxBlk(jj, n)] -= Dinv[ii].dot(F[jj].dot(xO[idxBlk(ii, n)]))
        if jj > 0:
            x[idxBlk(jj, n)] -= Dinv[ii].dot(E[jj].dot(xO[idxBlk(ii-1, n)]))

class CyclicSparseMatrix:
    def __init__(self, n, nblocks, E, D, F):
        # we want to make sure that the size of the blocks are good
        assert nblocks == len(E) and nblocks == len(D) and nblocks == len(F)
        self.n = n  # size of each block
        self.nblocks = nblocks
        self.E = E
        self.D = D
        self.F = F
        self.Dinv = []
        self.C = []

    def dot_slow(self, X):
        # dot product with a vector
        assert (X.shape[0] == self.n * self.nblocks)
        # we reshape it in order to be sure that it will be a 2D array
        X = np.reshape(X, (X.shape[0], -1))
        # we allocate the answer
        Ax = np.zeros(X.shape)

        for ii in range(self.nblocks):
            Ax[idxBlk(ii, self.n), :] += self.D[ii].dot(X[idxBlk(ii, self.n), :])
            if ii > 0:
                Ax[idxBlk(ii, self.n), :] += self.E[ii].dot(X[idxBlk(ii - 1, self.n), :])
            if ii < self.nblocks - 1:
                Ax[idxBlk(ii, self.n), :] += self.F[ii].dot(X[idxBlk(ii + 1, self.n), :])

        return Ax

    # this one is definitely faster
    def dot(self, X):
        # dot product with a vector
        assert (X.shape[0] == self.n * self.nblocks)
        # we reshape it in order to be sure that it will be a 2D array
        X = np.reshape(X, (X.shape[0], -1))
        # we allocate the answer
        Ax = np.zeros(X.shape)

        Ax[idxBlk(0, self.n), :] = self.D[0] @ X[idxBlk(0, self.n), :] + \
                                   self.F[0] @ X[idxBlk(0 + 1, self.n), :]

        for ii in range(1, self.nblocks-1):
            Ax[idxBlk(ii, self.n), :] = self.D[ii] @ X[idxBlk(ii, self.n), :] + \
                                        self.E[ii] @ X[idxBlk(ii - 1, self.n), :] +\
                                        self.F[ii] @ X[idxBlk(ii + 1, self.n), :]

        Ax[idxBlk(self.nblocks-1, self.n), :] = self.D[self.nblocks-1] @ X[idxBlk(self.nblocks-1, self.n), :] + \
                                       self.E[self.nblocks-1] @ X[idxBlk(self.nblocks-2, self.n), :]

        return Ax

    def factorize(self):
        # function to solve Cx = b, using cyclic reduction
        if len(self.Dinv) != 0:
            print("the matrix is already factorized")
        elif self.nblocks == 1:
            # if we are in a leaf, i.e. we are dealing with a dense matrix
            self.Dinv.append(la.inv(self.D[0].toarray()))
        else:
            # defing the indices
            IdxO = np.arange(0, self.nblocks, 2)
            IdxE = np.arange(1, self.nblocks, 2)

            # we start inverting the diagonal blocks
            Dinv = []

            for ii in range(len(IdxO)):
                Dinv.append(spla.factorized(self.D[IdxO[ii]]))

            E = []
            D = []
            F = []

            for ii in range(len(IdxE)):

                jj = IdxE[ii]
                Dlocal = np.zeros(self.D[jj].shape)

                Dlocal += self.D[jj]

                if jj > 0:  # this should never be false, but just in case
                    # update the diagonal
                    Dlocal -= self.E[jj].dot(Dinv[ii](self.F[jj - 1].toarray()))
                if jj < self.nblocks - 1:  # this can be true
                    # update the diagonal
                    Dlocal -= self.F[jj].dot(Dinv[ii + 1](self.E[jj + 1].toarray()))

                # now treating the off-diagonal blocks
                if jj > 0 and ii > 0:
                    Elocal = - self.E[jj].dot(Dinv[ii](self.E[jj - 1].toarray()))
                else:
                    # Elocal = spsp.csr_matrix((self.n, self.n), dtype=float)
                    Elocal = np.zeros((self.n, self.n))

                if jj < self.nblocks - 1 and ii < len(IdxE) - 1:
                    Flocal = - self.F[jj].dot(Dinv[ii + 1](self.F[jj + 1].toarray()))
                else:
                    # Flocal = spsp.csr_matrix((self.n, self.n), dtype=float)
                    Flocal = np.zeros((self.n, self.n))

                E.append(Elocal)
                D.append(Dlocal)
                F.append(Flocal)

            # defining the local inverses
            self.Dinv = Dinv
            # building the matrix for the resulting problem
            self.C = CyclicMatrix(self.n, len(IdxE), E, D, F)
            #factorizing the new matrix
            self.C.factorize()

    def solve_opt_c(self, b):
        assert self.n * self.nblocks == b.shape[0]
        # to be sure that b is a matrix

        b = np.reshape(b, (self.n * self.nblocks, -1))
        m = b.shape[1]

        if len(self.Dinv) == 0:
            print("Matrix was not factorized, we proceed to factorize it")
            self.factorize()


        # this is going to be called recursively
        if self.nblocks == 1:
            return self.Dinv[0](b)

        # function to solve Cx = b, using cyclic reduction
        IdxO = np.arange(0, self.nblocks, 2)
        IdxE = np.arange(1, self.nblocks, 2)

        # defining the local solutions
        xO = np.zeros((self.n * len(IdxO), m))
        # defining the right hand side for the
        # Schur complement
        bE = np.zeros((self.n * len(IdxE), m))
        # allocating the final answer
        x = np.zeros((self.n * self.nblocks, m))

        for ii in range(len(IdxO)):
            xO[idxBlk(ii, self.n), :] = self.Dinv[ii](b[idxBlk(IdxO[ii], self.n), :])

        for ii in range(len(IdxE)):
            jj = IdxE[ii]
            bE[idxBlk(ii, self.n), :] = b[idxBlk(jj, self.n), :]

            if jj > 0:  # this should never be false, but just in case
                # update the rhs
                bE[idxBlk(ii, self.n), :] -= self.E[jj].dot(xO[idxBlk(ii, self.n), :])
            if jj < self.nblocks - 1:  # this can be true
                # update the rhs
                bE[idxBlk(ii, self.n), :] -= self.F[jj].dot(xO[idxBlk(ii+1, self.n), :])

        xE = self.C.solve_opt_c(bE)

        for ii in range(len(IdxE)):
            jj = IdxE[ii]
            x[idxBlk(jj, self.n), :] = xE[idxBlk(ii, self.n), :]

        for ii in range(len(IdxO)):
            jj = IdxO[ii]
            x[idxBlk(jj, self.n), :] += xO[idxBlk(ii, self.n), :]
            if jj < self.nblocks - 1:
                x[idxBlk(jj, self.n), :] -= self.Dinv[ii](self.F[jj].dot(x[idxBlk(jj+1, self.n), :]))
            if jj > 0:
                x[idxBlk(jj, self.n), :] -= self.Dinv[ii](self.E[jj].dot(x[idxBlk(jj-1, self.n), :]))

        return x
