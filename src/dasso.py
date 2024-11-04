import numpy as np


def ADASSO(A, F, K, tau, err_ov):
    """
    ADASSO algorithm accelerated.

    Args:
        A (cupy.ndarray): (L, M) first input Boolean matrix in unipolar form {1,0}.
        F (cupy.ndarray): (L, N) second input Boolean matrix in unipolar form {1,0}.
        K (int): number of bases.
        tau (float): discriminative association confidence threshold.
        err_ov (float): tolerable overcoverage error ratio.
        gpu_id (int): The GPU device to use.

    Returns:
        B (cupy.ndarray): (L, K) basis Boolean matrix in unipolar form {1,0}.
        C (cupy.ndarray): (K, M) occurrence Boolean matrix corresponding to A in unipolar form {1,0}.
        D (cupy.ndarray): (K, N) occurrence Boolean matrix corresponding to F in unipolar form {1,0}.
        score (cupy.ndarray): (K, 1) optimization scores.
    """
    L, M = A.shape
    N = F.shape[1]

    B = np.zeros((L, K), dtype=int)
    C = np.zeros((K, M), dtype=int)
    score = np.zeros(K)
    D = np.zeros((K, N), dtype=int)

    w_ov = 1 / err_ov - 1

    # Weight for overcoverage error
    wA0 = w_ov * np.logical_not(A).astype(int)

    # Row sums for foreground (A) and background (F)
    a = np.maximum(np.sum(A, axis=1), 1)  # Ensure non-zero values in the denominator
    f = np.maximum(np.sum(F, axis=1), 1)
    # print(a)
    # print(a[:, None])
    # Calculate H matrix and related terms
    H = np.array((((A @ A.T) / a) - ((F @ F.T) / f)) >= tau).astype(float)
    # print(H[:10, :10])
    Hn1 = np.sum(H, axis=1)
    R = H.T @ F
    R[R - w_ov * (H.T @ np.logical_not(F).astype(int)) < 0] = 0
    Rsum = np.sum(R, axis=1) / N

    INDS = np.arange(L)

    for k in range(K):
        # Calculate the Q matrix
        Q = np.maximum(H.T @ (A - wA0), 0)
        arg = np.sum(Q, axis=1) / M - Rsum

        # Find the index with the maximum argument value
        ind = arg == np.max(arg)

        if np.sum(ind) > 1:
            ind = np.logical_and(Hn1 == np.min(Hn1[ind]), ind)
            j = np.min(INDS[ind])
        else:
            j = INDS[ind][0]

        # Update B, C, score, and D (if needed)
        B[:, k] = H[:, j]
        C[k, :] = (Q[j, :] > 0).astype(int)
        score[k] = arg[j]

        D[k, :] = (R[j, :] > 0).astype(int)

        # Update A for the next iteration
        A = np.logical_and(A, np.logical_not(np.outer(B[:, k], C[k, :]))).astype(int)

    return B, C, D, score


def SDASSO(A, F, K, tau, err_ov=0.5):
    """
    Symmetric Discriminative-Associative (SDASSO) algorithm.
    Args:
        A (numpy.ndarray): (L, M) first input Boolean matrix in unipolar form {1,0}.
        F (numpy.ndarray): (L, N) second input Boolean matrix in unipolar form {1,0}.
        K (int): number of bases.
        tau (float): discriminative association confidence threshold.
        err_ov (float, optional): tolerable overcoverage error ratio (default 0.5).

    Returns:
        B (numpy.ndarray): (L, K) basis Boolean matrix in unipolar form {1,0}.
        C (numpy.ndarray): (K, M) occurrence Boolean matrix corresponding to A in unipolar form {1,0}.
        D (numpy.ndarray): (K, N) occurrence Boolean matrix corresponding to F in unipolar form {1,0}.
        score (numpy.ndarray): (K, 1) optimization scores.
        in_ord (numpy.ndarray): (K, 1) indicates whether the bases come from the first input (1) or second input (2).
    """
    L, M = A.shape
    N = F.shape[1]

    # Initialize output variables
    B = np.zeros((L, K), dtype=int)
    C = np.zeros((K, M), dtype=int)
    D = np.zeros((K, N), dtype=int)
    score = np.zeros((K, 1))
    in_ord = np.ones((K, 1), dtype=int)

    # Run ADASSO for both directions
    B1, C1, D1, score1 = ADASSO(A, F, K, tau, err_ov)
    B2, D2, C2, score2 = ADASSO(F, A, K, tau, err_ov)

    # Combine results
    Bd = np.hstack((B1, B2))
    Cd = np.vstack((C1, C2))
    Dd = np.vstack((D1, D2))
    scored = np.hstack((score1, score2))

    # Select the best K bases
    for k in range(K):
        max_score_idx = np.nanargmax(
            scored
        )  # Find the index of the maximum score, ignoring NaN
        score[k] = scored[max_score_idx]

        # Update B, C, D based on max index
        B[:, k] = Bd[:, max_score_idx]
        C[k, :] = Cd[max_score_idx, :]
        D[k, :] = Dd[max_score_idx, :]

        # Mark the selected score as NaN so it's not selected again
        scored[max_score_idx] = np.nan

        # Determine if the base comes from the second input matrix (B2)
        if max_score_idx >= K:
            in_ord[k] = 2  # Comes from the second input matrix (B2)

    return B, C, D, score, in_ord
