import cupy as cp
from loguru import logger


def ADASSO(A, F, K, tau, err_ov, gpu_id=0):
    """
    ADASSO algorithm accelerated with CuPy for GPU usage.

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
    # Set the GPU device
    with cp.cuda.Device(gpu_id):
        A = cp.array(A)
        F = cp.array(F)
        logger.info(f"A: {A.shape}, F: {F.shape}")
        L, M = A.shape
        N = F.shape[1]

        # Initialize output variables on GPU
        B = cp.zeros((L, K), dtype=int)
        C = cp.zeros((K, M), dtype=int)
        score = cp.zeros(K)
        D = cp.zeros((K, N), dtype=int)

        w_ov = 1 / err_ov - 1

        # Weight for overcoverage error
        wA0 = w_ov * cp.logical_not(A).astype(int)

        # Row sums for foreground (A) and background (F)
        a = cp.maximum(
            cp.sum(A, axis=1), 1
        )  # Ensure non-zero values in the denominator
        f = cp.maximum(cp.sum(F, axis=1), 1)

        # Calculate H matrix and related terms
        H = cp.array(
            (((A @ A.T) / a[:, None]) - ((F @ F.T) / f[:, None])) >= tau, dtype=float
        )
        Hn1 = cp.sum(H, axis=1)
        R = H.T @ F
        R[R - w_ov * (H.T @ cp.logical_not(F).astype(int)) < 0] = 0
        Rsum = cp.sum(R, axis=1) / N

        INDS = cp.arange(L)

        for k in range(K):
            # Calculate the Q matrix
            Q = cp.maximum(H.T @ (A - wA0), 0)
            arg = cp.sum(Q, axis=1) / M - Rsum

            # Find the index with the maximum argument value
            ind = arg == cp.max(arg)

            if cp.sum(ind) > 1:
                ind = cp.logical_and(Hn1 == cp.min(Hn1[ind]), ind)
                j = cp.min(INDS[ind])
            else:
                j = INDS[ind][0]

            # Update B, C, score, and D (if needed)
            B[:, k] = H[:, j]
            C[k, :] = (Q[j, :] > 0).astype(int)
            score[k] = arg[j]

            D[k, :] = (R[j, :] > 0).astype(int)

            # Update A for the next iteration
            A = cp.logical_and(A, cp.logical_not(cp.outer(B[:, k], C[k, :]))).astype(
                int
            )

        return cp.asnumpy(B), cp.asnumpy(C), cp.asnumpy(D), cp.asnumpy(score)


def SDASSO(A, F, K, tau, err_ov=0.5, gpu_id=0):
    """
    Symmetric Discriminative-Associative (SDASSO) algorithm accelerated with CuPy for GPU usage.

    Args:
        A (cupy.ndarray): (L, M) first input Boolean matrix in unipolar form {1,0}.
        F (cupy.ndarray): (L, N) second input Boolean matrix in unipolar form {1,0}.
        K (int): number of bases.
        tau (float): discriminative association confidence threshold.
        err_ov (float, optional): tolerable overcoverage error ratio (default 0.5).
        gpu_id (int): The GPU device to use.

    Returns:
        B (cupy.ndarray): (L, K) basis Boolean matrix in unipolar form {1,0}.
        C (cupy.ndarray): (K, M) occurrence Boolean matrix corresponding to A in unipolar form {1,0}.
        D (cupy.ndarray): (K, N) occurrence Boolean matrix corresponding to F in unipolar form {1,0}.
        score (cupy.ndarray): (K, 1) optimization scores.
        in_ord (cupy.ndarray): (K, 1) indicates whether the bases come from the first input (1) or second input (2).
    """
    # Set the GPU device
    with cp.cuda.Device(gpu_id):
        A = cp.array(A)
        F = cp.array(F)

        L, M = A.shape
        N = F.shape[1]

        # Initialize output variables on GPU
        B = cp.zeros((L, K), dtype=int)
        C = cp.zeros((K, M), dtype=int)
        D = cp.zeros((K, N), dtype=int)
        score = cp.zeros((K, 1))
        in_ord = cp.ones((K, 1), dtype=int)

        # Run ADASSO for both directions
        B1, C1, D1, score1 = ADASSO(A, F, K, tau, err_ov, gpu_id)
        B2, D2, C2, score2 = ADASSO(F, A, K, tau, err_ov, gpu_id)

        # Combine results
        Bd = cp.hstack((B1, B2))
        Cd = cp.vstack((C1, C2))
        Dd = cp.vstack((D1, D2))
        scored = cp.hstack((score1, score2))

        # Select the best K bases
        for k in range(K):
            max_score_idx = cp.nanargmax(
                scored
            )  # Find the index of the maximum score, ignoring NaN
            score[k] = scored[max_score_idx]

            # Update B, C, D based on max index
            B[:, k] = Bd[:, max_score_idx]
            C[k, :] = Cd[max_score_idx, :]
            D[k, :] = Dd[max_score_idx, :]

            # Mark the selected score as NaN so it's not selected again
            scored[max_score_idx] = cp.nan

            # Determine if the base comes from the second input matrix (B2)
            if max_score_idx >= K:
                in_ord[k] = 2  # Comes from the second input matrix (B2)

        return (
            cp.asnumpy(B),
            cp.asnumpy(C),
            cp.asnumpy(D),
            cp.asnumpy(score),
            cp.asnumpy(in_ord),
        )
