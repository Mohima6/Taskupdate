import os
import numpy as np
from itertools import combinations
from collections import Counter

DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\Taskupdate"
CONN_METRIC = "wPLI"
CONN_BASE = os.path.join(DATASET_PATH, "connectivity", CONN_METRIC)
OUTPUT_FEATURES = os.path.join(DATASET_PATH, "machine_learning", "features")
os.makedirs(OUTPUT_FEATURES, exist_ok=True)

FREQUENCY_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

def load_and_average_subject(band_dir, subject_base, metric_lower):
    epoch_files = [f for f in os.listdir(band_dir)
                   if f.startswith(subject_base) and f.endswith(f"_{metric_lower}.npy")]
    if not epoch_files:
        return None
    matrices = [np.load(os.path.join(band_dir, f)) for f in epoch_files]
    return np.mean(matrices, axis=0)

for band in FREQUENCY_BANDS:
    print(f"\nProcessing {band}...")
    band_dir = os.path.join(CONN_BASE, band)
    if not os.path.exists(band_dir):
        print(f"  Folder {band_dir} not found")
        continue

    label_files = [f for f in os.listdir(band_dir) if f.endswith("_labels.npy")]
    if not label_files:
        continue

    subject_base_names, labels_list = [], []
    for lf in label_files:
        base = lf.replace("_labels.npy", "")
        labels = np.load(os.path.join(band_dir, lf))
        if len(labels) > 0:
            subject_base_names.append(base)
            labels_list.append(labels[0])

    subject_matrices, valid_labels = [], []
    for base, lab in zip(subject_base_names, labels_list):
        mat = load_and_average_subject(band_dir, base, CONN_METRIC.lower())
        if mat is not None:
            subject_matrices.append(mat)
            valid_labels.append(lab)

    if len(subject_matrices) == 0:
        continue

    # handle inconsistent channel counts
    shapes = [m.shape[0] for m in subject_matrices]
    if len(set(shapes)) > 1:
        common_n = Counter(shapes).most_common(1)[0][0]
        filtered = [(m, l) for m, l in zip(subject_matrices, valid_labels) if m.shape[0] == common_n]
        subject_matrices, valid_labels = zip(*filtered)
        subject_matrices, valid_labels = list(subject_matrices), list(valid_labels)

    n_channels = subject_matrices[0].shape[0]
    pairs = list(combinations(range(n_channels), 2))
    n_features = len(pairs)

    X = np.zeros((len(subject_matrices), n_features))
    for i, mat in enumerate(subject_matrices):
        for f, (p, q) in enumerate(pairs):
            X[i, f] = mat[p, q]
    y = np.array(valid_labels)

    np.save(os.path.join(OUTPUT_FEATURES, f"{CONN_METRIC}_{band}_X.npy"), X)
    np.save(os.path.join(OUTPUT_FEATURES, f"{CONN_METRIC}_{band}_y.npy"), y)
    print(f"  Saved {X.shape[0]} subjects, {n_features} features")