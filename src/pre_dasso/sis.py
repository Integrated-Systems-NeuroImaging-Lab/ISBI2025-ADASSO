import numpy as np
from loguru import logger
from scipy.linalg import svd
from scipy.stats import ks_2samp

from ..utils import elbow_point


class SourceInformedSegmentation:
    def __init__(
        self,
        eeg,
        win_ref,
        win_dcs,
        win_slid,
        win_step,
        win_overlap,
        domain,
        ref_avg_remove,
    ):
        """Implement of the algorithm "Source Informed Segmentation (ONLY IN TIME DOMIAN for now)"

        Args:
            eeg (np.ndarray): 2d array (n_chan, timepoints)
            win_ref (int):    Initial Reference Window (suggest 20 samples)
            win_dcs (int):    Decision Window (suggest 20 samples)
            win_slid (int): Sliding Winodw (default = win_ref)
            win_step (int): Step (default = 1)
            win_overlap (int): Overlap (<= min(win_ref, win_slid) ,default = 0)
            domain (string): "time"
            ref_avg_remove (bool): Reference Space Average Removal (time domain only)
        """

        # pass parameters
        self.domain = domain
        if self.domain == "time":
            self.data = eeg
        self.win_ref = win_ref
        self.win_dcs = win_dcs
        self.win_slid = win_slid
        self.win_step = win_step
        self.win_overlap = win_overlap
        self.ref_avg_remove = ref_avg_remove

        # init
        self.n_chan, self.ts = eeg.shape
        logger.debug(f"shape of eeg: ({self.n_chan}, {self.ts})")

        self.init_params_seg_searching()

    def ks_test(self):
        """Quantify the difference between two data blocks Z and X by KS test, also mapping the them into feature space F.
        Args:
            Z (np.ndarray): Reference Block
            X (np.ndarray): Sliding Block
            F (np.ndarray): Feature Space from SVD

        Returns:
            decs(bool): the flag of difference. True for p < 0.05 (diff); False for p >= 0.05 (same).
            pval(float): P Value.
        """
        err_ref = np.sum(np.abs(self.Z) ** 2, axis=0) - np.sum(
            np.abs(self.F.T @ self.Z) ** 2, axis=0
        )
        err_sld = np.sum(np.abs(self.X) ** 2, axis=0) - np.sum(
            np.abs(self.F.T @ self.X) ** 2, axis=0
        )
        ks_result = ks_2samp(err_ref, err_sld)
        """ Null Hypothesis (H0): The two samples come from the same distribution.
            Alternative Hypothesis (H1): The two samples come from different distributions.

            When you perform the kstest2, it returns a p-value.
            The p-value indicates the probability of observing the test results under the null hypothesis.

            P-value < 0.05: This typically indicates that there is less than a 5% probability that the observed differences between the two samples are due to chance. Therefore, you reject the null hypothesis and conclude that the two samples likely come from different distributions.
            P-value ≥ 0.05: This suggests that you do not have enough evidence to reject the null hypothesis, meaning the two samples could plausibly come from the same distribution.

            Interpretation:
            If you get a p-value < 0.05 from kstest2, it suggests that the difference between the two samples is statistically significant, leading to the conclusion that the two samples are likely drawn from different distributions.
            """
        # logger.debug(ks_result)
        self.pval = ks_result.pvalue
        self.diff = True if self.pval < 0.05 else False

    def generate_blocks(
        self,
    ):
        """Generate Block Z, X, F by updated indics."""
        self.Z = self.data[
            :, self.segment_points[-1] : self.segment_points[-1] + self.cur_seg_size
        ]

        self.slid_pnt = self.segment_points[-1] + self.cur_seg_size - self.win_overlap
        self.X = self.data[:, self.slid_pnt : self.slid_pnt + self.win_slid]
        # logger.debug(
        #     f"Z:[{self.segment_points[-1]}:{self.segment_points[-1] + self.cur_seg_size}], X:[{self.slid_pnt}:{self.slid_pnt + self.win_slid}]"
        # )

        if np.sum(np.abs(self.Z)) == 0:
            self.idx_elbow = 1
            self.energy_rate = 1
            self.singular_val = 0
            self.F = np.ones((self.data.shape[0], 1)) / np.sqrt(self.data.shape[0])
        else:
            # * Step 3: Calculate the SVD of reference block Z.
            Ur, self.singular_val, _ = svd(self.Z, full_matrices=False)
            # logger.debug(f"singular values: {Sr}")
            # * Step 4 Take the 1st singular vectors down to the elbow as the estimated feature subspaece{U_R_hat} -> F
            self.idx_elbow, self.energy_rate = elbow_point(self.singular_val)
            self.F = Ur[:, : self.idx_elbow]

        if self.ref_avg_remove and self.domain == "time":
            self.Z = self.Z - np.mean(self.Z, axis=0, keepdims=True)

    def alter_hypo_DIFF(self):
        self.omega_d += 1
        if self.omega_d < self.win_dcs:
            # * Save possible boundary and its p-value
            if self.pval < self.min_pval:
                self.min_pval = self.pval
                self.slid_pnt_min_pval = self.slid_pnt
            # * W = W + 1
            self.cur_seg_size += self.win_step
        else:
            self.segment_points.append(self.slid_pnt_min_pval)
            self.init_params_seg_searching()

    def null_hypo_SAME(self):
        if self.omega_d >= self.win_dcs:
            self.segment_points.append(self.slid_pnt_min_pval)
            self.init_params_seg_searching()
        else:
            self.omega_d = 0
            self.cur_seg_size += self.win_step

    def init_params_seg_searching(self):
        self.cur_seg_size = self.win_ref
        self.omega_d = 0
        self.min_pval = 1000.0
        self.slid_pnt_min_pval = 0

    def compute(self):
        # reset history of segment points in each computation.
        self.segment_points = [0]
        # check whether the reference block excede the boundary of length of data
        while (
            self.segment_points[-1]
            + self.cur_seg_size
            + self.win_slid
            - self.win_overlap
            <= self.ts
        ):
            self.generate_blocks()
            self.ks_test()
            if self.diff:
                self.alter_hypo_DIFF()
            else:
                self.null_hypo_SAME()
        self.segment_points.append(self.data.shape[1] - 1)
        logger.info(f"segment points found: {self.segment_points}")

        return self.segment_points
