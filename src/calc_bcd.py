import itertools
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from loguru import logger

from .dasso_cp import ADASSO
from .utils import get_available_gpus, split_list


def calculate_and_save_ADASSO(A_list, F_list, K, tau, err_ov, save_path, gpu_id):
    """
    Calculate ADASSO for each pair of A and F from their respective HDF5 files and save the results.

    Args:
        A_list (list): List of file paths for A HDF5 files.
        F_list (list): List of file paths for F HDF5 files.
        K (int): Target number of components (Rank).
        tau (float): Threshold for forming the boolean number.
        err_ov (float): Punishing factor for over-coverage.
        save_path (str or Path): Directory path where results should be saved.

    Raises:
        AssertionError: If the lengths of A_list and F_list do not match.
    """
    # Ensure that A_list and F_list have the same length
    assert len(A_list) == len(
        F_list
    ), "A_list and F_list must have the same number of files."
    logger.info(f"A_list:{A_list}")
    logger.info(f"F_list:{F_list}")
    # Convert save_path to a Path object if it's not already
    save_path = Path(save_path)

    # Ensure save_path exists
    save_path.mkdir(parents=True, exist_ok=True)

    # Traverse through each pair of A and F files
    for a_file, f_file in zip(A_list, F_list):
        logger.info(f"Processing A: {a_file} and F: {f_file}")

        # Read A and F matrices from their respective HDF5 files
        with h5py.File(a_file, "r") as a_h5, h5py.File(f_file, "r") as f_h5:
            A = a_h5["bool_mat"][:]
            F = f_h5["bool_mat"][:]

        # Ensure A and F have the same shape for processing
        assert (
            A.shape[0] == F.shape[0]
        ), f"A and F matrices must have the same shape. Got {A.shape} and {F.shape}."

        # Perform ADASSO on the current pair
        B, C, D, score = ADASSO(A, F, K, tau, err_ov, gpu_id)
        logger.debug(f"B: {np.unique(B,return_counts=True)}")
        logger.debug(f"score: {score}")
        # Prepare the output file name based on the input file names and parameters
        a_filename = Path(a_file).stem  # Get the stem (filename without extension) of A
        f_filename = Path(f_file).stem  # Get the stem (filename without extension) of F
        output_filename = (
            f"A_{a_filename}_F_{f_filename}_K{K}_tau{tau}_errOv{err_ov}.h5"
        )
        output_filepath = save_path / output_filename  # Create the full file path

        # Save the result matrices B, C, D to a new HDF5 file
        with h5py.File(output_filepath, "w") as result_h5:
            result_h5.create_dataset("B", data=B)
            result_h5.create_dataset("C", data=C)
            result_h5.create_dataset("D", data=D)
            result_h5.create_dataset("score", data=score)

        logger.info(f"Results saved to {output_filepath}")


def get_A_F_files(base_dir, subject_range, task_A, task_F):
    if not isinstance(base_dir, Path):
        base_dir = Path(base_dir)

    # Ensure the subject numbers are formatted as two digits (e.g., "01" to "09")
    subject_list = [f"{subject:02d}" for subject in subject_range]

    # List all HDF5 files in the base directory
    all_files = list(base_dir.glob("*.h5"))

    # Create file lists for A (left task) and F (right task)
    A_list = []
    F_list = []

    # Traverse through all the files and filter by subject number and task
    for file in all_files:
        filename = file.stem  # Get the filename without extension (AXX_YYYYY)
        parts = filename.split("_")

        if len(parts) != 2:
            continue  # Skip files that don't match the "AXX_YYYYY" pattern

        subject_number = parts[0][1:]  # Extract XX (subject number)
        task_name = parts[1]  # Extract YYYYY (task name)

        # Check if the file belongs to the specified subject range
        if subject_number in subject_list:
            if task_name == task_A:
                A_list.append(file)
            elif task_name == task_F:
                F_list.append(file)

    # Sort the lists to ensure subjects are in order
    A_list.sort()
    F_list.sort()

    # Ensure that we have matching numbers of A and F files
    assert len(A_list) == len(
        F_list
    ), "The number of left task files and right task files do not match!"

    return A_list, F_list


if __name__ == "__main__":
    # bool_mat_dir = "./data/bool_matrix/BCI_IV2a/2dr_w30_sfreq_150"
    bool_mat_dir = "./data/bool_matrix/BCI_IV2a/2dr_eseg10_sfreq_150"
    bool_mat_dir = Path(bool_mat_dir)
    save_dir = (
        Path("./data/BCD")
        / bool_mat_dir.parts[-2]
        / f"{bool_mat_dir.parts[-1]}_{datetime.now().strftime('%y%m%d%H%M')}"
    )
    tau = [0.08, 0.06, 0.04, 0.02]
    err_ov = [0.5, 0.4, 0.3, 0.2]
    # tau = [0.5, 0.1, 0.05, 0.01, 0.008, 0.006, 0.004, 0.002]
    # err_ov = [0.5, 0.4, 0.3, 0.2, 0.1]
    grid_search = list(itertools.product(tau, err_ov))
    logger.info(f"have {len(grid_search)} combanation of grid search")
    for tau, err_ov in grid_search:
        logger.debug(f"tau: {tau}; err_ov: {err_ov}")
        common_kwargs = {
            "K": 20,
            "tau": tau,
            "err_ov": err_ov,
            "save_path": save_dir,
        }

        subject_range = range(1, 10)  # Subjects 1 to 9
        A_list_left, F_list_right = get_A_F_files(
            base_dir=bool_mat_dir,
            subject_range=subject_range,
            task_A="left",
            task_F="right",
        )
        A_list_right, F_list_left = get_A_F_files(
            base_dir=bool_mat_dir,
            subject_range=subject_range,
            task_A="right",
            task_F="left",
        )
        A_list = A_list_left + A_list_right
        F_list = F_list_right + F_list_left
        logger.info(f"A file list: {A_list}")
        logger.info(f"F file list: { F_list}")

        processes = []
        # FIX:
        gpus = get_available_gpus()
        # gpus = list(range(8))

        num_gpus = len(gpus)
        A_splits = split_list(A_list, num_gpus)
        F_splits = split_list(F_list, num_gpus)

        logger.info(common_kwargs)
        for i, gpu_id in enumerate(gpus):
            processes.append(
                mp.Process(
                    target=calculate_and_save_ADASSO,
                    kwargs={
                        **common_kwargs,
                        "A_list": A_splits[i],
                        "F_list": F_splits[i],
                        "gpu_id": gpu_id,
                    },
                )
            )

        for p in processes:
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()
    logger.info("BDF processes have completed.")
