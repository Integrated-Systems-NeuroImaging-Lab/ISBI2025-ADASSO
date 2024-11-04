from pathlib import Path
from typing import List

import mne
import numpy as np
import scipy
from loguru import logger
from mne.preprocessing import ICA


class PreprocessSingleEEGMMIDB:
    def __init__(self, path_raw_edf, ica_exclude_chnl, tmin=-0.1, tmax=0.9):
        self.path_raw_edf = path_raw_edf
        self.tmin = tmin
        self.tmax = tmax
        self.raw = mne.io.read_raw_edf(
            input_fname=path_raw_edf,
            preload=True,
        )
        self.ica_exclude_chnl = ica_exclude_chnl
        self.ica = ICA(max_iter="auto", random_state=2470, method="picard")

        self.preprocess()

    def montage_setting_eegmmidb(self):
        # montage setting
        chnl_names = [i.replace(".", "") for i in self.raw.info["ch_names"]]
        dict_chnl_rename = dict(zip(self.raw.info["ch_names"], chnl_names))
        dict_chnl_rename["T9.."] = "P9"
        dict_chnl_rename["T10."] = "P10"

        self.raw.rename_channels(mapping=dict_chnl_rename)
        self.raw.set_montage(montage="biosemi64", match_case=False)

    def preprocess(self):
        self.montage_setting_eegmmidb()
        # filtering
        self.raw.notch_filter(freqs=60.0)
        self.raw.filter(l_freq=1.0, h_freq=50.0)
        # ICA
        self.ica.fit(self.raw)
        self.ica.exclude = self.ica_exclude_chnl
        self.ica.apply(self.raw)

    def get_raw(self):
        return self.raw

    def epoch_and_save(self, path_fif):
        events = mne.events_from_annotations(raw=self.raw)
        len_events = len(events[1])
        if len_events > 1:
            epochs = mne.Epochs(
                raw=self.raw,
                events=events[0],
                event_id=events[1],
                tmin=self.tmin,
                tmax=self.tmax,
                preload=True,
            )
            epochs.save(fname=path_fif, overwrite=True)
        else:
            self.raw.save(fname=path_fif, overwrite=True)


class PreprocessEEGMMIDB:
    def __init__(
        self,
        path_data: str = "./data/",
        path_save: str = "./data/preprocessed_eegmmidb",
        id_subject: List[int] = list(range(1, 110)),
        id_runs: List[int] = list(range(1, 15)),
    ):
        """Preprocess the EEGMMIDB dataset by given subjects and runs (default: all)
        and save the epochs to .npy file, preprocessed edf file to .fif file.

        Args:
            path_data: path to the root of eegbci (default: ./data/)
            path_save: path to save the preprocessed data (default: ./data/preprocessed_eegmmidb)
            id_subject: list of chosen subjects (default: 1-109)
            id_runs: list of chosen runs (default: 1-14)
        """

        self.id_subjects = id_subject
        self.id_runs = id_runs
        self.path_save = Path(path_save)
        self.path_data = path_data

    def preprocess_and_save(self):
        for s in self.id_subjects:
            path_edfs = mne.datasets.eegbci.load_data(
                subject=s, runs=self.id_runs, path=self.path_data
            )
            path_subject_root = self.path_save / Path(*Path(path_edfs[0]).parts[-2:-1])
            path_subject_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"process and save to {str(path_subject_root.resolve())}")
            for p in path_edfs:
                path_epo_fif = path_subject_root / Path(
                    *Path(p).parts[-1:]
                ).with_suffix(".fif")
                path_epo_fif = path_epo_fif.with_name(
                    path_epo_fif.stem + "-epo" + path_epo_fif.suffix
                )
                # TODO: replace the ica setting with toml config
                preprocess_single_eegmmidb = PreprocessSingleEEGMMIDB(
                    path_raw_edf=p, ica_exclude_chnl=[0, 1, 2]
                )
                preprocess_single_eegmmidb.epoch_and_save(path_epo_fif)
                logger.info(f"{path_epo_fif} saved.")
                # break
                #


class PreprocessBCI_IV2a_JZ:
    def __init__(self, path_data: str, path_save: str, resample: float = 250.0):
        self.path_data = Path(path_data)
        self.path_save = Path(path_save)
        self.path_save.mkdir(parents=True, exist_ok=True)
        logger.info(f"Seaching .mat file in: {self.path_data.resolve()}")
        self.path_mat_files = list(self.path_data.resolve().glob("*.mat"))
        logger.debug(f".mat file found: {self.path_mat_files}")

        # montage preparation
        montage = mne.channels.make_standard_montage("standard_1020")
        selected_channels = [
            "Fz",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "P1",
            "Pz",
            "P2",
            "POz",
        ]
        positions = montage.get_positions()["ch_pos"]
        selected_channel_positions = {
            ch: positions[ch] for ch in selected_channels if ch in positions
        }
        self.montage_bci42a_22c = mne.channels.make_dig_montage(
            ch_pos=selected_channel_positions, coord_frame="head"
        )
        self.dict_label = {"left": 1, "right": 2, "feet": 3, "tongue": 4}
        self.sfreq = 250.0
        self.resample = resample
        self.info = mne.create_info(
            ch_names=selected_channels, sfreq=self.sfreq, ch_types="eeg"
        )
        self.info.set_montage(self.montage_bci42a_22c)
        self.epoch_sec = 3.0

        cls_mat = set([p.stem.split("_")[-1] for p in self.path_mat_files])
        logger.info(f"{len(cls_mat)} classes found: {cls_mat}")

        for p_mat in self.path_mat_files:
            mat_epoch = self.preprocess_mat_iv2a(p_mat)
            name_fif = p_mat.stem + "-epo.fif"
            path_fif = self.path_save / name_fif
            mat_epoch.save(fname=path_fif, overwrite=True)
            logger.info(f"{path_fif} saved")

    def preprocess_mat_iv2a(self, path_file):
        label = path_file.stem.split("_")[-1]
        mat_data = scipy.io.loadmat(file_name=path_file.resolve())
        data = mat_data[label]
        data = np.transpose(data, (0, 2, 1))
        # events
        event_id = {label: self.dict_label[label]}
        Y = np.ones(mat_data[label].shape[0]) * event_id[label]
        eventLength = Y.shape[0]
        ev = [i * self.sfreq * self.epoch_sec for i in range(eventLength)]
        events = np.column_stack(
            (
                np.array(ev, dtype=int),
                np.zeros(eventLength, dtype=int),
                np.array(Y, dtype=int),
            )
        )
        epochs = mne.EpochsArray(
            data=data, info=self.info, events=events, event_id=event_id
        )
        epochs.filter(l_freq=0.5, h_freq=50)
        epochs.resample(sfreq=self.resample)
        return epochs


if __name__ == "__main__":
    # # preprocess EEGMMIDB to epo.fif
    # p_eegmmidb = PreprocessEEGMMIDB(
    #     path_data="../data/",
    #     path_save="../data/preprocessed_eegmmidb",
    #     id_subject=list(range(16, 110)),
    # )
    # p_eegmmidb.preprocess_and_save()

    p_bciiv2a_jz = PreprocessBCI_IV2a_JZ(
        path_data="./data/jiazhen/Data_BCI4_2a",
        path_save="./data/preprocessed_BCI_IV2a/sfreq_150",
        resample=150.0,
    )
