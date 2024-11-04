import numpy as np
from loguru import logger
from tqdm import tqdm

from .csp import csp


def generate_A_F(
    L,
    K,
    N,
    M,
    prob_flip,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Primary process for A and F (Boolean matrices)
    E = (np.random.rand(L, K) > 0.5).astype(int)
    G = (np.random.rand(L, K) > 0.5).astype(int)
    X = (np.random.rand(K, M) > 0.5).astype(int)
    Y = (np.random.rand(K, N) > 0.5).astype(int)

    R_a = np.sum(X, axis=1) / M
    R_f = np.sum(Y, axis=1) / N

    A_primary = ((E @ X) > 0).astype(int)
    F_primary = ((G @ Y) > 0).astype(int)

    SN_a = np.random.rand(*A_primary.shape) < prob_flip
    SN_f = np.random.rand(*F_primary.shape) < prob_flip

    A = np.where(SN_a, 1 - A_primary, A_primary)
    F = np.where(SN_f, 1 - F_primary, F_primary)

    return A, F, E, G, X, Y, R_a, R_f


def calculate_B_gt(E, G, R_a, R_f, V_n):
    """
    Calculates the ground-truth discriminative basis matrix B_gt based on the overlap
    between basis vectors from matrices E and G, and recurrence rate arrays R_a and R_f.

    Parameters:
    - E (ndarray): Basis matrix E, shape (L, Ka).
    - G (ndarray): Basis matrix G, shape (L, Kf).
    - R_a (ndarray): Recurrence rate array for each basis vector in A, shape (Ka,).
    - R_f (ndarray): Recurrence rate array for each basis vector in F, shape (Kf,).
    - V_n (int): Number of basis vectors in G associated with each basis vector in E.

    Returns:
    - B_gt (ndarray): Ground-truth basis matrix, shape (L, Ka).
    """
    L, Ka = E.shape
    _, Kf = G.shape
    B_gt = np.zeros((L, Ka), dtype=bool)

    for i in range(Ka):
        overlap_vectors = E[:, i]

        # Determine the range of associated basis vectors in G
        start_idx = i * V_n
        end_idx = min(start_idx + V_n, Kf)

        for j in range(start_idx, end_idx):
            # Apply the condition based on corresponding R_a[i] and R_f[j] values
            if R_f[j] >= R_a[i]:
                overlap_vectors = np.logical_and(
                    overlap_vectors, np.logical_not(G[:, j])
                )
            else:
                overlap_vectors = np.logical_and(
                    overlap_vectors, np.ones(L, dtype=bool)
                )

        B_gt[:, i] = overlap_vectors

    return B_gt


def calculate_coverage(B, C, B_gt, X):
    """
    Calculate the coverage percentage based on Equation (23).

    Parameters:
    - B: np.array, Boolean matrix output from ADASSO (predicted basis vectors)
    - C: np.array, recurrence pattern matrix associated with B
    - B_gt: np.array, ground-truth discriminative basis matrix
    - X: np.array, recurrence pattern matrix associated with B_gt

    Returns:
    - coverage_percentage: float, the coverage metric in percentage
    """
    B_C = (B @ C > 0).astype(int)
    B_gt_X = (B_gt @ X > 0).astype(int)
    intersection = np.logical_and(B_C, B_gt_X).astype(int)

    numerator = np.linalg.norm(x=intersection, ord=1)
    denominator = np.linalg.norm(x=B_gt_X, ord=1)

    if denominator == 0:
        coverage_percentage = 0.0  # Avoid division by zero
    else:
        coverage_percentage = numerator / denominator

    return coverage_percentage


if __name__ == "__main__":
    # Parameters for the test
    seed = 2470

    # for K in tqdm(range(100, 110, 10)):
    #     L = 1000  # Number of rows (sources)
    #     logger.info(f"K: {K}")
    #     M = 1000
    #     N = M
    #     max_coverage = 0
    #     for V_n in range(1, 6):
    #         for prob_flip in (0, 0.55, 0.05):
    #             for err_ov in (0.5, 0.25, 0.125, 0.0625, 0.01):
    #                 for tau in (0.1, 1.0, 0.1):
    #                     # Generate A, F, and the components
    #                     A, F, E, G, X, Y, R_a, R_f = generate_A_F(
    #                         L=L, M=M, N=N, K=K, prob_flip=prob_flip
    #                     )
    #                     # logger.info(f"Matrix A: { np.unique(A, return_counts=True)}")
    #                     # logger.info(f"Matrix F: { np.unique(F, return_counts=True)}")
    #                     # continue
    #                     B_gt = calculate_B_gt(E, G, R_a, R_f, V_n)
    #                     # logger.info(f"B_gt: {np.unique(B_gt, return_counts=True)}")

    #                     B, C, D, socre = ADASSO(A=A, F=F, K=K, tau=tau, err_ov=err_ov)
    #                     # logger.info(f"coverage: {calculate_coverage(B=B,C=C,B_gt=B_gt,X=X)}")

    #                     coverage = calculate_coverage(B=B, C=C, B_gt=B_gt, X=X)
    #                     if coverage > max_coverage:
    #                         max_coverage = coverage

    #     logger.info(f"max cov in K={K}: {max_coverage}")

    for K in tqdm(range(10, 110, 10)):
        L = 1000  # Number of rows (sources)
        logger.info(f"K: {K}")
        M = 1000
        N = M
        max_coverage = 0
        for V_n in range(1, 6):
            for prob_flip in (0, 0.55, 0.05):
                # for err_ov in (0.5, 0.25, 0.125, 0.0625, 0.01):
                # Generate A, F, and the components
                A, F, E, G, X, Y, R_a, R_f = generate_A_F(
                    L=L, M=M, N=N, K=K, prob_flip=prob_flip
                )
                # logger.info(f"Matrix A: { np.unique(A, return_counts=True)}")
                # logger.info(f"Matrix F: { np.unique(F, return_counts=True)}")
                # continue
                B_gt = calculate_B_gt(E, G, R_a, R_f, V_n)
                # logger.info(f"B_gt: {np.unique(B_gt, return_counts=True)}")

                W_selected, features_class1, features_class2 = csp(
                    X1=A[np.newaxis, ...], X2=F[np.newaxis, ...], num_filters=K
                )
                bool_Ws = (W_selected > 0).astype(int)

                occur_C = (W_selected.T @ (A - (np.ones_like(A) - A) * 100) > 0).astype(
                    int
                )
                logger.debug(f"occur_C: {np.unique(occur_C,return_counts=True)}")
                # exit()

                coverage = calculate_coverage(B=bool_Ws, C=occur_C, B_gt=B_gt, X=X)
                logger.info(f"coverage: {coverage}")
                if coverage > max_coverage:
                    max_coverage = coverage

        logger.info(f"max cov in K={K}: {max_coverage}")
