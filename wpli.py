import os
import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm

DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\Taskupdate"
BAND_DECOMP_DIR = os.path.join(DATASET_PATH, "band_decomposition")
OUTPUT_BASE = os.path.join(DATASET_PATH, "connectivity", "wPLI")

FREQUENCY_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

for band in FREQUENCY_BANDS:
    os.makedirs(os.path.join(OUTPUT_BASE, band), exist_ok=True)


def compute_wpli(signal1, signal2):
    """
    Weighted Phase Lag Index (wPLI)

    wPLI = |E(Im(X))| / E(|Im(X)|)

    X = analytic_signal1 * conj(analytic_signal2)
    """

    analytic1 = hilbert(signal1)
    analytic2 = hilbert(signal2)

    imag_cross = np.imag(
        analytic1 * np.conj(analytic2)
    )

    numerator = np.abs(
        np.mean(imag_cross)
    )

    denominator = np.mean(
        np.abs(imag_cross)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def compute_connectivity_matrix(epoch_data):

    n_channels, _ = epoch_data.shape

    mat = np.eye(n_channels, dtype=np.float32)

    for i in range(n_channels):
        for j in range(i + 1, n_channels):

            val = compute_wpli(
                epoch_data[i],
                epoch_data[j]
            )

            mat[i, j] = val
            mat[j, i] = val

    return mat


print("Starting wPLI computation...")

for band in FREQUENCY_BANDS:

    band_input_dir = os.path.join(BAND_DECOMP_DIR, band)
    band_output_dir = os.path.join(OUTPUT_BASE, band)

    epoch_files = [
        f for f in os.listdir(band_input_dir)
        if f.endswith(".npy")
    ]

    label_dir = os.path.join(DATASET_PATH, "processed_data_epochs")

    for epoch_file in tqdm(epoch_files, desc=f"wPLI {band}"):

        band_data = np.load(
            os.path.join(band_input_dir, epoch_file)
        )

        n_epochs = band_data.shape[0]

        base_name = epoch_file.replace(
            f"_{band}.npy",
            ""
        )

        label_file = os.path.join(
            label_dir,
            f"{base_name}_labels.npy"
        )

        if not os.path.exists(label_file):
            continue

        labels = np.load(label_file)

        for ep_idx in range(n_epochs):

            conn_mat = compute_connectivity_matrix(
                band_data[ep_idx]
            )

            np.save(
                os.path.join(
                    band_output_dir,
                    f"{base_name}_epoch{ep_idx:03d}_wpli.npy"
                ),
                conn_mat
            )

        np.save(
            os.path.join(
                band_output_dir,
                f"{base_name}_labels.npy"
            ),
            labels
        )

print("wPLI computation completed.")