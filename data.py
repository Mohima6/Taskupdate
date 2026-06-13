import os
import numpy as np
import mne
from tqdm import tqdm
from scipy.stats import zscore

# ========== CONFIGURATION ==========
DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\Taskupdate"
OUTPUT_FOLDER = os.path.join(DATASET_PATH, "processed_data_epochs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

LOW_FREQ = 0.1
HIGH_FREQ = 70
NOTCH_FREQ = 50

EPOCH_LENGTH = 5  # seconds
OVERLAP = 0.5  # 50% overlap -> 2.5 s step

# ========== FIND ALL EDF FILES ==========
edf_files = []
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(".edf"):
            edf_files.append(os.path.join(root, file))

print(f"Found {len(edf_files)} EDF files\n")


# ========== ICA HELPER (FastICA only, no extra dependencies) ==========
def apply_ica(raw, n_components=0.99, random_state=42):
    """Fit FastICA and remove EOG artifacts if an EOG channel exists."""
    # Identify EOG channels (case‑insensitive)
    eog_channels = [ch for ch in raw.ch_names if 'eog' in ch.lower() or 'eye' in ch.lower()]
    if not eog_channels:
        print("   No EOG channel found. ICA will be fitted but no components will be excluded.")

    # Use FastICA (always available via sklearn)
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method='fastica',
        random_state=random_state,
        max_iter='auto'
    )

    ica.fit(raw)

    if eog_channels:
        # Use the first EOG channel to find artefactual components
        eog_idx, scores = ica.find_bads_eog(raw, ch_name=eog_channels[0])
        ica.exclude = eog_idx
        print(f"   Excluding {len(eog_idx)} EOG components")
    else:
        ica.exclude = []

    cleaned = raw.copy()
    ica.apply(cleaned)
    return cleaned


# ========== MAIN PREPROCESSING FUNCTION ==========
def preprocess_file(file_path):
    file_name = os.path.basename(file_path)
    print(f"\nProcessing: {file_name}")

    # ----- 1. Load -----
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.pick('eeg')  # keep only EEG channels
    print(f"   EEG channels: {len(raw.ch_names)}")

    # ----- 2. Filter (bandpass + notch) -----
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design='firwin', verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)

    # ----- 3. ICA (artifact removal) -----
    raw = apply_ica(raw)

    # ----- 4. Epoching (5 s, 50% overlap) -----
    step = EPOCH_LENGTH * (1 - OVERLAP)  # 2.5 s
    overlap = EPOCH_LENGTH - step  # 2.5 s
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=EPOCH_LENGTH,
        overlap=overlap,
        preload=True,
        verbose=False
    )
    epoch_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    print(f"   Epochs generated: {epoch_data.shape[0]}")

    # ----- 5. Z‑score normalisation (per epoch, per channel, across time) -----
    epoch_data = zscore(epoch_data, axis=-1)
    epoch_data = np.nan_to_num(epoch_data)  # replace NaN/Inf with 0

    # ----- 6. Label extraction from filename -----
    base_name = os.path.splitext(file_name)[0].lower()
    if base_name.startswith('h'):
        label = 0  # healthy control
    elif base_name.startswith('mdd'):
        label = 1  # depressed (MDD)
    else:
        print(f"   Unknown label in filename: {file_name}, skipping.")
        return

    labels = np.full(len(epoch_data), label, dtype=np.uint8)

    # ----- 7. Storage optimisation (float16 for data, uint8 for labels) -----
    epoch_data = epoch_data.astype(np.float16)

    # ----- 8. Save to output folder -----
    out_prefix = os.path.join(OUTPUT_FOLDER, os.path.splitext(file_name)[0])
    np.save(f"{out_prefix}_epochs.npy", epoch_data)
    np.save(f"{out_prefix}_labels.npy", labels)

    print(f"   Saved: {out_prefix}_epochs.npy (shape {epoch_data.shape}, dtype float16)")
    print(f"   Saved: {out_prefix}_labels.npy (shape {labels.shape}, dtype uint8)")


# ========== PROCESS ALL FILES ==========
for fpath in tqdm(edf_files, desc="Overall progress"):
    preprocess_file(fpath)

print(f"\n All processing finished. Clean epochs saved in:\n {OUTPUT_FOLDER}")