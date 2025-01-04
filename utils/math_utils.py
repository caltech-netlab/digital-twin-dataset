import itertools
import numpy as np
import cvxpy as cp
np.set_printoptions(linewidth=250)


def matrix2block(M, dim):
    """
    :param M: np.ndarray, shape=(dim*n, dim*n), we have n x n of dim x dim blocks .
    :param dim: int, dimension of each block.
    :returns: np.ndarray, shape=(dim, dim, n, n), list of blocks.
    """
    assert (
        (len(M.shape) == 2) and (M.shape[0] % dim == 0) and (M.shape[1] % dim == 0)
    ), M.shape
    m, n = M.shape[0] // dim, M.shape[1] // dim
    return M.reshape(m, dim, n, dim).transpose(0, 2, 1, 3)


def block2matrix(A):
    """
    :param A: np.ndarray, shape=(M, N, m, n). Each block is of shape (m, n).
    """
    assert len(A.shape) == 4, A.shape
    M, N, m, n = A.shape
    return A.transpose(0, 2, 1, 3).reshape(M * m, N * n)


def permute_matrix(A, perm):
    return A[perm][:, perm]


def block_permute_matrix(A, perm, W):
    """
    Operation is not in-place.
    :param A: np.ndarray, shape=(3m, 3m), matrix to be permuted.
    :param perm: list of int, permutation of indices.
        Note that the indices are block indices, not row/column indices.
        Input perm will be the block labels of the output matrix.
    :returns: np.ndarray, shape=(3m, 3m), permuted matrix.
    """
    assert (
        (len(A.shape) == 2) and (A.shape[0] == A.shape[1]) and (A.shape[0] % W == 0)
    ), A.shape
    assert (len(perm) == (A.shape[0] / W)) and (
        list(range(int(A.shape[0] / W))) == sorted(perm)
    ), perm
    return matrix2block(A, W)[perm][:, perm].transpose(0, 2, 1, 3).reshape(A.shape)


def kron_reduction(Y, reduced_idx):
    """
    Given a Y matrix, Kron reduce the reduced_idx by computing the Schur complement.
    :param Y: np.ndarray, square matrix.
    :param reduced_idx: list of int, the indices to be reduced.
        Note that the matrix is not indexed by blocks.
    """
    assert (len(Y.shape) == 2) and (Y.shape[0] == Y.shape[1]), Y.shape
    N = Y.shape[0]
    p = N - len(reduced_idx)
    perm = [i for i in range(N) if i not in reduced_idx] + list(reduced_idx)
    Y = permute_matrix(Y, perm)
    A, B, C, D = Y[:p, :p], Y[:p, p:], Y[p:, :p], Y[p:, p:]
    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        raise RuntimeError("kron_reduction: D is singular.\nY:\n{}\nD:\n{}".format(Y, D))
    return A - B @ D_inv @ C


def remove_row_column(Y, remove_idx):
    assert (len(Y.shape) == 2) and (Y.shape[0] == Y.shape[1]), Y.shape
    inds = np.array(list(set(range(Y.shape[0])) - set(remove_idx)), dtype=int)
    return Y[inds][:, inds]


def matrix_equivalent_up_to_permutation(A, B, W, skip=[], atol=1e-5, rtol=1e-5):
    """
    Check if two matrices are equivalent up to permutation of blocks.
    :param A: np.ndarray, shape=(m, m), matrix to be compared.
    :param B: np.ndarray, shape=(m, m), matrix to be compared.
    :param W: int, block size.
    :param skipfirst: list of int, do not check permutation for these indices.
        i.e. The two matrices must match for these rows/columns.
    :returns: bool, True if A is equivalent to B up to permutation of blocks.
    """
    assert (
        (len(A.shape) == 2) and (A.shape[0] == A.shape[1]) and (A.shape == B.shape)
    ), (A.shape, B.shape)
    m = A.shape[0] // W
    nonskip = [i for i in range(m) if i not in skip]
    max_err = np.inf
    B_perm_best = None
    for nonskip_perm in itertools.permutations(nonskip):
        nonskip_perm = list(nonskip_perm)[::-1]
        perm = [i if i in skip else nonskip_perm.pop() for i in range(m)]
        B_perm = block_permute_matrix(B, perm, W)
        err = np.abs(A - B_perm).max()
        max_err, B_perm_best = (
            (err, B_perm) if err < max_err else (max_err, B_perm_best)
        )
        if np.allclose(A, B_perm, atol=atol, rtol=rtol):
            return True, 0, None
    return False, max_err, B_perm_best


if __name__ == "__main__":
    pass
