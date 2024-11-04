import re
import time
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tomllib  # Python 3.11+
from kneed import KneeLocator
from loguru import logger


def split_list(lst, n):
    """
    Splits a list `lst` into `n` roughly equal parts.
    """
    # Calculate the size of each chunk
    k, m = divmod(len(lst), n)

    # Return a list of chunks
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def load_settings(path_to_settings: str):
    """
    Load settings from a TOML file with error handling.

    Args:
        path_to_settings (str): The path to the TOML configuration file.

    Returns:
        dict: The parsed TOML file as a Python dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        tomllib.TOMLDecodeError: If the TOML file is improperly formatted.
    """
    try:
        with open(path_to_settings, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        logger.error(f"Settings file not found: {path_to_settings}")
        raise
    except tomllib.TOMLDecodeError:
        logger.error(f"Error decoding TOML file: {path_to_settings}")
        raise


# Load settings into a global variable
SETTINGS = load_settings("./config/setting.toml")


class DatasetHandler:
    """
    Base class for handling datasets.

    Each subclass must implement the get_epoch_paths method to handle specific datasets.
    """

    def __init__(self, path_fif: str, id_subject: int, runs: List[int], anno: str):
        """
        Initialize the dataset handler.

        Args:
            path_fif (str): Path to the directory containing .fif files.
            id_subject (int): ID of the subject.
            runs (List[int]): List of run numbers for the dataset.
            anno (str): Annotation to filter the epochs.
        """
        self.path_fif = path_fif
        self.id_subject = id_subject
        self.runs = runs
        self.anno = anno

    def get_epoch_paths(self) -> List[Path]:
        """
        Abstract method to retrieve epoch paths.

        This method should be implemented by subclasses to return a list of .fif file paths.

        Returns:
            List[Path]: List of paths to the .fif files.

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _epo_files(self, pattern: str) -> List[Path]:
        """
        Helper method to match .fif files based on a regular expression pattern.

        Args:
            pattern (str): The regex pattern to match the file names.

        Returns:
            List[Path]: List of matched file paths.
        """
        return [
            p
            for p in Path(self.path_fif).rglob("*-epo.fif")
            if re.match(pattern, p.stem)
        ]


class EEGMMIDBDatasetHandler(DatasetHandler):
    """
    Handler for EEGMMIDB dataset.

    This class handles the retrieval of epoch paths for the EEGMMIDB dataset.
    """

    def get_epoch_paths(self) -> List[Path]:
        """
        Retrieve the epoch file paths for the EEGMMIDB dataset.

        Filters the files based on the subject ID and run number.

        Returns:
            List[Path]: List of paths to the filtered .fif files.
        """
        pattern = rf"S{self.id_subject:03}R(\d{{2}})"
        matched_files = self._epo_files(pattern)
        return [
            p
            for p in matched_files
            if int(re.match(pattern, p.stem).group(1)) in self.runs
        ]


class BCI_IV2aDatasetHandler(DatasetHandler):
    """
    Handler for BCI_IV2a dataset.

    This class handles the retrieval of epoch paths for the BCI_IV2a dataset.
    """

    def get_epoch_paths(self) -> List[Path]:
        """
        Retrieve the epoch file paths for the BCI_IV2a dataset.

        Filters the files based on the subject ID and task annotation.

        Returns:
            List[Path]: List of paths to the filtered .fif files.
        """
        pattern = rf"A{self.id_subject:02}[ET]_(tongue|left|right|feet)"
        matched_files = self._epo_files(pattern)
        return [
            p for p in matched_files if re.match(pattern, p.stem).group(1) == self.anno
        ]


def get_dataset_handler(
    name_dataset: str, path_fif: str, id_subject: int, runs: List[int], anno: str
) -> DatasetHandler:
    """
    Factory method to select the appropriate dataset handler.

    Args:
        name_dataset (str): The name of the dataset (e.g., "EEGMMIDB" or "BCI_IV2a").
        path_fif (str): Path to the directory containing .fif files.
        id_subject (int): Subject ID.
        runs (List[int]): List of run numbers for the dataset.
        anno (str): Annotation to filter the epochs.

    Returns:
        DatasetHandler: An instance of the appropriate dataset handler class.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    handlers = {
        "EEGMMIDB": EEGMMIDBDatasetHandler,
        "BCI_IV2a": BCI_IV2aDatasetHandler,
    }
    if name_dataset in handlers:
        return handlers[name_dataset](path_fif, id_subject, runs, anno)
    raise ValueError(f"Unsupported dataset: {name_dataset}")


def extract_epoch_fif(
    name_dataset: str, name_setting: str, path_fif: str, id_subject: int
) -> Tuple[List[str], str]:
    """
    Extract the .fif epoch file paths for a given dataset.

    Uses the settings from a TOML file to determine the annotation and run numbers.

    Args:
        name_dataset (str): The name of the dataset (e.g., "EEGMMIDB" or "BCI_IV2a").
        name_setting (str): The specific setting/task within the dataset.
        path_fif (str): Path to the directory containing .fif files.
        id_subject (int): Subject ID.

    Returns:
        Tuple[List[str], str]: A tuple containing a list of file paths and the annotation.

    Raises:
        KeyError: If the dataset or setting is not found in the configuration.
    """
    try:
        setting = SETTINGS[name_dataset][name_setting]
    except KeyError as e:
        raise KeyError(f"{e} is not in {list(SETTINGS[name_dataset].keys())}")

    logger.info(
        f"Collecting .fif (of epochs) with annotation {setting['anno']} and runs {setting['runs']}"
    )

    handler = get_dataset_handler(
        name_dataset, path_fif, id_subject, setting["runs"], setting["anno"]
    )
    path_epochs_fif = handler.get_epoch_paths()
    if not path_epochs_fif:
        raise FileNotFoundError(f"No *-epo.fif found for subject {id_subject}")

    return path_epochs_fif, setting["anno"]


def elbow_point(x: np.ndarray, method="second_derivative") -> Tuple[int, float]:
    """
    Calculate the elbow point of the singular value list from SVD using different methods.

    Args:
        x (np.ndarray): Array of singular values.
        method (str): The method to use for finding the elbow point. Options are "second_derivative", "first_derivative", or "kneedle".

    Returns:
        Tuple[int, float]: The index of the elbow point and the energy ratio up to the elbow.
    """
    methods = {
        "second_derivative": _second_derivative_method,
        "first_derivative": _first_derivative_method,
        "kneedle": _kneedle_method,
    }

    if method not in methods:
        logger.warning(f"Method '{method}' not recognized. Defaulting to 'kneedle'.")
        method = "kneedle"

    idx = methods[method](x)
    total_energy = np.sum(x**2)
    energy_ratio = np.sum(x[:idx] ** 2) / total_energy
    while energy_ratio < 0.8:
        idx += 2
        energy_ratio = np.sum(x[:idx] ** 2) / total_energy

    return idx, energy_ratio


def _second_derivative_method(x: np.ndarray) -> int:
    """
    Find the elbow point using the second derivative method.

    Args:
        x (np.ndarray): Array of singular values.

    Returns:
        int: The index of the elbow point.
    """
    dx2 = np.diff(np.diff(x))
    idx = np.argmax(dx2) + 1
    # logger.debug(f"2nd derivative method: {dx2}")
    return idx


def _first_derivative_method(x: np.ndarray) -> int:
    """
    Find the elbow point using the first derivative method.

    Args:
        x (np.ndarray): Array of singular values.

    Returns:
        int: The index of the elbow point.
    """
    dx1 = np.diff(x)
    idx = np.argmax(dx1) + 1
    # logger.debug(f"1st derivative method: {dx1}")
    return idx


def _kneedle_method(x: np.ndarray) -> int:
    """
    Find the elbow point using the KneeLocator method from the kneed library.

    Args:
        x (np.ndarray): Array of singular values.

    Returns:
        int: The index of the elbow point.
    """
    kneedle = KneeLocator(
        x=list(range(len(x))), y=x, curve="convex", direction="decreasing"
    )
    idx = kneedle.elbow
    logger.debug(f"kneedle method at {idx}")
    return idx


def get_available_gpus():
    """
    Check which GPUs are available (no processes running).
    Returns a list of GPU IDs that are free.
    """
    # Run the nvidia-smi command and get the output
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
    )
    output = result.stdout.decode("utf-8").strip().split("\n")

    available_gpus = []
    for line in output:
        gpu_id, memory_used = line.split(",")
        gpu_id = int(gpu_id.strip())
        memory_used = int(memory_used.strip())

        # If memory_used is 10MB, the GPU is free
        if memory_used < 10:
            available_gpus.append(gpu_id)
    logger.info(f"available gpu: {available_gpus}")
    return available_gpus


def check_gpu_availability():
    """
    Function to check which GPUs are currently in use based on running processes.
    Returns a set of GPU IDs that are actively being used.
    """
    try:
        # Get all processes running on the GPUs with GPU IDs
        output = (
            subprocess.check_output(
                "nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader",
                shell=True,
            )
            .decode("utf-8")
            .strip()
        )
        active_gpus = set()  # Track GPUs with running processes
        if output:
            # Parse the output to get the GPU IDs with processes
            for line in output.splitlines():
                _, gpu_bus_id = line.split(",")
                gpu_id = (
                    subprocess.check_output(
                        f"nvidia-smi --query-gpu=index --format=csv,noheader --id={gpu_bus_id.strip()}",
                        shell=True,
                    )
                    .decode("utf-8")
                    .strip()
                )
                active_gpus.add(int(gpu_id))
        return active_gpus
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running nvidia-smi: {e}")
        return set()


def check_available_gpus(required_gpus, interval=1, timeout=None):
    """
    Function that continuously checks the availability of a specified number of GPUs.
    Logs changes in availability and returns True once the required number of GPUs are free.

    Args:
    - required_gpus (int): The number of GPUs that need to be available.
    - interval (int): The time interval (in seconds) between checks. Default is 1 second.
    - timeout (int): Optional timeout in seconds. If provided, the function will stop after this time and return False.

    Returns:
    - bool: True if the required number of GPUs become available, False if a timeout occurs (if timeout is set).
    """
    start_time = time.time()
    output = (
        subprocess.check_output(
            "nvidia-smi --query-gpu=count --format=csv,noheader", shell=True
        )
        .decode("utf-8")
        .strip()
    )
    num_gpus = int(output.splitlines()[0].strip())

    previous_available_gpus = set()  # To track changes in availability

    while True:
        # Get the active GPUs
        active_gpus = check_gpu_availability()

        # Calculate available GPUs by subtracting active GPUs from total GPUs
        available_gpus = set(range(num_gpus)) - active_gpus

        # Log only if there is a change in availability
        if available_gpus != previous_available_gpus:
            logger.info(f"Available GPUs have changed: {available_gpus}")
            previous_available_gpus = available_gpus

        # Check if the required number of GPUs are available
        if len(available_gpus) >= required_gpus:
            logger.info(f"Required {required_gpus} GPUs are available.")
            return True

        # If a timeout is specified, check if it has been exceeded
        if timeout and (time.time() - start_time) > timeout:
            logger.warning(
                f"Timeout exceeded. Only {len(available_gpus)} GPUs available, but {required_gpus} are required."
            )
            return False

        # Wait for the specified interval before checking again
        time.sleep(interval)
