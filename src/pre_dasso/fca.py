import numpy as np
from loguru import logger
from mne import (
    compute_covariance,
    make_forward_solution,
    read_trans,
    setup_volume_source_space,
)
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import (
    apply_inverse_epochs,
    make_inverse_operator,
)
from scipy.linalg import svd

from ..utils import elbow_point


class FunctionalConnectivityAnalysis:
    def __init__(self, epoch, path_fs="./data/", inverse_method="sLORETA"):
        self.epoch = epoch
        self.fs_dir = fetch_fsaverage(subjects_dir=path_fs, verbose=50)

        self.inverse_method = inverse_method

    def apply_inverse_to_source(self):
        # if not isinstance(self.epoch, BaseEpochs):
        #     noise_cov = compute_covariance(epochs=self.epoch)
        # else:
        #     raise ValueError(
        #         f"Unsupported data type: {type(self.epoch)}. Please provide either Epochs data."
        #     )
        # * using pos=12.12 to determine 1024 points in source space.
        src = setup_volume_source_space(
            "fsaverage",
            pos=12.12,
            mri=f"{self.fs_dir}/mri/brain.mgz",
            bem=f"{self.fs_dir}/bem/fsaverage-5120-5120-5120-bem-sol.fif",
            subjects_dir=self.fs_dir,
            add_interpolator=False,
        )
        # Compute the forward solution
        fwd = make_forward_solution(
            self.epoch.info,
            trans=read_trans(f"{self.fs_dir}/bem/fsaverage-trans.fif"),
            src=src,
            bem=f"{self.fs_dir}/bem/fsaverage-5120-5120-5120-bem-sol.fif",
            meg=False,
            eeg=True,
        )
        noise_cov = compute_covariance(epochs=self.epoch)
        # Create the inverse operator
        inverse_operator = make_inverse_operator(
            self.epoch.info, fwd, noise_cov=noise_cov, loose="auto", depth=0.8
        )
        # Compute the source estimate using sLORETA
        self.epoch.set_eeg_reference(projection=True)
        list_stc = apply_inverse_epochs(
            self.epoch, inverse_operator, lambda2=1.0 / 9.0, method="sLORETA"
        )
        return list_stc

    @staticmethod
    def compute_connectivity(S):
        # * follow the naming rule of Ali's Paper
        S_hat = S - np.mean(S, axis=1, keepdims=True)
        U, D, _ = svd(S_hat, full_matrices=False)
        idx_D, energy_ratio = elbow_point(D)
        U_R = U[:, :idx_D]
        D_R = D[:idx_D]
        cortical_maps = U_R * D_R
        return cortical_maps, S_hat, idx_D, energy_ratio
