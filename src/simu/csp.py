import numpy as np
from scipy.linalg import eigh


def compute_covariance_matrix(trial):
    """Compute normalized covariance matrix for a single trial."""
    cov_matrix = np.dot(trial, trial.T) / np.trace(np.dot(trial, trial.T))
    return cov_matrix


def average_covariance_matrices(data):
    """Compute the average covariance matrix across all trials in a dataset."""
    cov_matrices = [compute_covariance_matrix(trial) for trial in data]
    return np.mean(cov_matrices, axis=0)


def compute_features(data, W_selected):
    features = []
    for trial in data:
        projected = np.dot(W_selected.T, trial)
        # Calculate log-variance of each projected component
        log_variance = np.log(np.var(projected, axis=1))
        features.append(log_variance)
    return np.array(features)


def csp(X1, X2, num_filters):
    """
    Perform CSP on two classes of EEG data.

    Parameters:
        X1 (numpy array): EEG data for Class 1 with shape (num_trials, num_channels, num_samples)
        X2 (numpy array): EEG data for Class 2 with shape (num_trials, num_channels, num_samples)
        num_filters (int): Number of spatial filters to select (e.g., 2 or 4)

    Returns:
        W_selected (numpy array): Selected spatial filters (shape: [num_channels, num_filters])
        features_class1 (numpy array): Features for Class 1 trials (shape: [num_trials, num_filters])
        features_class2 (numpy array): Features for Class 2 trials (shape: [num_trials, num_filters])
    """
    # Step 1: Compute the mean covariance matrices for each class
    C1_mean = average_covariance_matrices(X1)
    C2_mean = average_covariance_matrices(X2)

    # Step 2: Composite covariance matrix
    C_composite = C1_mean + C2_mean

    C1_mean += np.eye(C1_mean.shape[0]) * 1e-5
    C_composite += np.eye(C_composite.shape[0]) * 1e-5

    # Step 3: Solve the generalized eigenvalue problem
    eigenvalues, eigenvectors = eigh(C1_mean, C_composite)

    # Step 4: Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, sorted_indices]

    # Step 5: Select the spatial filters
    W_first = W[:, : num_filters // 2]
    W_last = W[:, -num_filters // 2 :]
    W_selected = np.hstack((W_first, W_last))

    # Step 6: Project data onto spatial filters and compute features
    features_class1 = compute_features(X1, W_selected)
    features_class2 = compute_features(X2, W_selected)

    return W_selected, features_class1, features_class2
