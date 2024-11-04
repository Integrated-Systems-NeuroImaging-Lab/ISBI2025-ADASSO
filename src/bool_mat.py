from asyncio import timeout
import datetime
import multiprocessing as mp
from pathlib import Path
import time

import h5py
import mne
import numpy as np
import tomllib
from loguru import logger

from .pre_dasso.boolz_cp import booleanization
from .pre_dasso.fca import FunctionalConnectivityAnalysis
from .pre_dasso.sis import SourceInformedSegmentation
from .utils import (
    extract_epoch_fif,
    get_available_gpus,
    split_list,
    check_available_gpus,
)


def evenly_split(len_list, num_pieces):
    # Calculate the step size to divide the list into num_pieces parts
    step = len_list / num_pieces
    # Calculate the breakpoints
    breakpoints = [round(step * i) for i in range(1, num_pieces)]
    return [0] + breakpoints + [len_list]


def compute_bool_matrix_and_save(
    name_dataset: str,
    setting: str,
    id_subject: int,
    fifs_dir: str,
    win_size: int,
    save_dir: Path,
    gpu_id: int,
    n_evenly_split=None,
):
    # Extract epochs from .fif files
    path_epochs_fif, anno = extract_epoch_fif(
        name_dataset=name_dataset,
        name_setting=setting,
        path_fif=fifs_dir,
        id_subject=id_subject,
    )

    # Handle case with no epochs found
    if not path_epochs_fif:
        raise FileNotFoundError(f"No *-epo.fif found for subject {id_subject}")

    logger.info(f"epo.fif files found: {path_epochs_fif}")

    # Prepare HDF5 file for saving data
    match name_dataset:
        case "EEGMMIDB":
            path_h5 = save_dir / f"S{id_subject:03}_{setting}.h5"
        case "BCI_IV2a":
            path_h5 = save_dir / f"A{id_subject:02}_{setting}.h5"
        case _:
            raise ValueError(
                f"output file naming rule of {name_dataset} has not been created!"
            )

    with h5py.File(path_h5, "a") as f_h5:
        # Initialize HDF5 datasets only once for efficiency
        is_first_segment = True

        for fifs_dir in path_epochs_fif:
            logger.info(f"Processing {fifs_dir} ...")
            epoch = mne.read_epochs(fname=fifs_dir)[anno]
            epoch_data = epoch.get_data()

            # Run Functional Connectivity Analysis (FCA)
            fca = FunctionalConnectivityAnalysis(epoch=epoch)
            stcs = fca.apply_inverse_to_source()

            for idx, (eeg_data, stc) in enumerate(zip(epoch_data, stcs)):
                logger.info(f"Processing Epoch {idx+1}/{len(epoch_data)} ...")

                if not n_evenly_split:
                    # * Perform Source Informed Segmentation (SIS)
                    sis = SourceInformedSegmentation(
                        eeg=eeg_data,
                        win_ref=win_size,
                        win_dcs=win_size,
                        win_slid=win_size,
                        win_step=1,
                        win_overlap=0,
                        domain="time",
                        ref_avg_remove=False,
                    )
                    segment_points = sis.compute()
                else:
                    segment_points = evenly_split(
                        len_list=eeg_data.shape[1], num_pieces=n_evenly_split
                    )

                logger.info(
                    f"Segment points: {segment_points} ({len(segment_points)-1} segments)"
                )

                # Initialize containers for boolean matrix and lengths
                bool_mat = []
                bool_len = []

                # Process each segment
                for s in range(1, len(segment_points)):
                    logger.info(f"Segment [{segment_points[s-1]}, {segment_points[s]})")

                    # Compute connectivity
                    cortical_maps, S_hat, r_rank, energy_ratio = (
                        fca.compute_connectivity(
                            S=stc.data[:, segment_points[s - 1] : segment_points[s]]
                        )
                    )

                    if r_rank == 0:
                        logger.warning("0 rank segment")
                        bool_len.append(0)
                        continue
                    logger.debug(
                        f"rank:{r_rank}/{segment_points[s]-segment_points[s-1]}, energy_ratio: {energy_ratio:.3f}"
                    )
                    # * Perform Booleanization
                    threshold = booleanization(
                        segment=S_hat,
                        R=r_rank,
                        list_alpha=[0.05],
                        n_simu=500,
                        need_norm=False,
                        gpu_id=gpu_id,
                        chunk_size=270,
                    )

                    # Create boolean cortical map

                    r_bool_cot_map = np.hstack(
                        [
                            (
                                cortical_maps[:, r][:, None] < threshold[:, r][:, None]
                            ).astype(int)
                            for r in range(r_rank)
                        ]
                    )

                    bool_len.append(r_rank)
                    # Append the boolean cortical map
                    bool_mat.append(r_bool_cot_map)

                # Combine boolean matrices across segments
                if bool_mat:
                    bool_mat = np.hstack(bool_mat)

                logger.debug(np.unique(bool_mat, return_counts=True))
                # Save to HDF5
                if is_first_segment:
                    # Create datasets for the first segment
                    f_h5.create_dataset(
                        "bool_mat", data=bool_mat, maxshape=(bool_mat.shape[0], None)
                    )
                    f_h5.create_dataset(
                        "segments", data=segment_points, maxshape=(None,)
                    )
                    f_h5.create_dataset("bool_len", data=bool_len, maxshape=(None,))
                    is_first_segment = False
                else:
                    # Resize and append new data
                    f_h5["bool_mat"].resize(
                        (
                            bool_mat.shape[0],
                            f_h5["bool_mat"].shape[1] + bool_mat.shape[1],
                        )
                    )
                    f_h5["bool_mat"][:, -bool_mat.shape[1] :] = bool_mat

                    f_h5["segments"].resize(
                        (f_h5["segments"].shape[0] + len(segment_points),)
                    )
                    f_h5["segments"][-len(segment_points) :] = segment_points

                    f_h5["bool_len"].resize(
                        (f_h5["bool_len"].shape[0] + len(bool_len),)
                    )
                    f_h5["bool_len"][-len(bool_len) :] = bool_len

                logger.info(
                    f"Updated HDF5: bool_mat: {f_h5['bool_mat'].shape}, "
                    f"segments: {f_h5['segments'].shape}, "
                    f"bool_len: {f_h5['bool_len'].shape}"
                )
                logger.info(f"write to {path_h5}")


if __name__ == "__main__":
    # * log setting
    log_dir = Path("./results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_log = log_dir / f"{now_time}.log"
    logger.add(path_log.resolve(), rotation="500 MB", retention="10 days")
    mne.set_log_level(verbose="ERROR")

    # * EEGMMIDB S001-S109 >>>
    # name_dataset = "EEGMMIDB"
    # win_size = 30
    # subjects = list(range(1, 110))
    # fifs_dir = f"./data/preprocessed_{name_dataset}"
    # save_dir = Path(f"./data/bool_matrix/{name_dataset}/2dr_w{win_size}/")
    # save_dir.mkdir(parents=True, exist_ok=True)
    # * EEGMMIDB S001-S109 <<<

    # * BCI_IV2a >>>
    name_dataset = "BCI_IV2a"
    win_size = 30
    subjects = list(range(1, 10))
    fifs_dir = f"./data/preprocessed_{name_dataset}/sfreq_150"
    # save_dir = Path(f"./data/bool_matrix/{name_dataset}/2dr_w{win_size}_sfreq_150/")
    n_evenly_split = 5
    save_dir = Path(
        f"./data/bool_matrix/{name_dataset}/2dr_eseg{n_evenly_split}_sfreq_150/"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    # * BCI_IV2a <<<

    with open("./config/setting.toml", "rb") as f:
        setting = tomllib.load(f)

    required_gpus = 8
    timeout_wait = 2e4
    if not check_available_gpus(
        required_gpus=required_gpus, interval=10, timeout=timeout_wait
    ):
        # FIX: manually set gpus
        gpus = list(range(0, 4))
        logger.warning(
            f"cannot have {required_gpus} gpus in {timeout_wait/60/60:2f} h, using gpu{gpus}"
        )
    else:
        gpus = get_available_gpus()

    processes = []
    num_gpus = len(gpus)
    if num_gpus == 0:
        raise RuntimeError("None of the GPUs are available!")
    subject_list = split_list(subjects, num_gpus)
    logger.info(f"split jobs into {subject_list}")

    def _wrapper_compute_bool_mat(subject_list, gpu_id):
        timer_start = time.perf_counter()
        for id_subject in subject_list:
            timer_subject = time.perf_counter()
            logger.info(f"[gpu {gpu_id}] running for subject {id_subject}")
            for k in list(setting[name_dataset].keys()):
                compute_bool_matrix_and_save(
                    name_dataset=name_dataset,
                    setting=k,
                    id_subject=id_subject,
                    fifs_dir=fifs_dir,
                    win_size=win_size,
                    save_dir=save_dir,
                    n_evenly_split=n_evenly_split,
                    gpu_id=gpu_id,
                )
            logger.info(
                f"[gpu {gpu_id}] subject {id_subject} take {(time.perf_counter()-timer_subject)/60:.2f}(m)."
            )
        logger.info(
            f"[gpu {gpu_id}] completed in {(time.perf_counter()-timer_start)/60:.2f}(m)."
        )

    for i, gpu_id in enumerate(gpus):
        processes.append(
            mp.Process(
                target=_wrapper_compute_bool_mat,
                kwargs=dict(
                    subject_list=subject_list[i],
                    gpu_id=gpu_id,
                ),
            )
        )

    for p in processes:
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    logger.info("all Bool mat processes have completed.")
