import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from itertools import combinations
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')
DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\Taskupdate"
CONN_METRIC = "wPLI"
CONN_BASE = os.path.join(DATASET_PATH, "connectivity", CONN_METRIC)
RESULT_BASE = os.path.join(DATASET_PATH, "machine_learning", "results", CONN_METRIC)
os.makedirs(RESULT_BASE, exist_ok=True)
FREQUENCY_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
RANDOM_STATE = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
classifiers = {
    "LogisticRegression": {
        "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "params": {"C": [0.01, 0.1, 1, 10, 100]}
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "params": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
    },
    "SVM": {
        "model": SVC(random_state=RANDOM_STATE, probability=True),
        "params": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["rbf", "linear"]}
    }
}
try:
    from xgboost import XGBClassifier
    classifiers["XGBoost"] = {
        "model": XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
        "params": {"n_estimators": [50, 100], "max_depth": [3, 6], "learning_rate": [0.01, 0.1]}
    }
    print("XGBoost available.\n")
except ImportError:
    print("XGBoost not installed – skipping.\n")
def specificity_score_func(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0
def load_and_average_subject(band_dir, subject_base, metric_lower):
    epoch_files = [f for f in os.listdir(band_dir)
                   if f.startswith(subject_base) and f.endswith(f"_{metric_lower}.npy")]
    if not epoch_files:
        return None
    matrices = [np.load(os.path.join(band_dir, f)) for f in epoch_files]
    return np.mean(matrices, axis=0)
for band in FREQUENCY_BANDS:
    print(f"\n{'='*50}\n{CONN_METRIC} - {band}\n{'='*50}")
    band_dir = os.path.join(CONN_BASE, band)
    if not os.path.exists(band_dir):
        print(f"  Folder not found: {band_dir}")
        continue
    label_files = [f for f in os.listdir(band_dir) if f.endswith("_labels.npy")]
    if not label_files:
        print("  No label files found.")
        continue
    subject_base_names = []
    labels_list = []
    for lf in label_files:
        base = lf.replace("_labels.npy", "")
        labels = np.load(os.path.join(band_dir, lf))
        if len(labels) > 0:
            subject_base_names.append(base)
            labels_list.append(labels[0])
    subject_matrices = []
    valid_labels = []
    for base, lab in zip(subject_base_names, labels_list):
        mat = load_and_average_subject(band_dir, base, CONN_METRIC.lower())
        if mat is not None:
            subject_matrices.append(mat)
            valid_labels.append(lab)
    if len(subject_matrices) == 0:
        print("  No valid subject matrices.")
        continue
    shapes = [m.shape[0] for m in subject_matrices]
    if len(set(shapes)) > 1:
        common_n = Counter(shapes).most_common(1)[0][0]
        filtered = [(m, l) for m, l in zip(subject_matrices, valid_labels) if m.shape[0] == common_n]
        subject_matrices, valid_labels = zip(*filtered)
        subject_matrices, valid_labels = list(subject_matrices), list(valid_labels)
        print(f"  Inconsistent channels – kept {common_n} channels, {len(subject_matrices)} subjects.")
    else:
        common_n = shapes[0]
    n_channels = common_n
    n_subjects = len(subject_matrices)
    n_healthy = valid_labels.count(0)
    n_mdd = valid_labels.count(1)
    print(f"  Subjects: {n_subjects} (Healthy: {n_healthy}, MDD: {n_mdd})")
    pairs = list(combinations(range(n_channels), 2))
    n_features = len(pairs)
    X = np.zeros((n_subjects, n_features))
    for i, mat in enumerate(subject_matrices):
        for f, (p, q) in enumerate(pairs):
            X[i, f] = mat[p, q]
    y = np.array(valid_labels)
    print(f"  Features: {n_features} (from {n_channels} channels)")
    band_result_dir = os.path.join(RESULT_BASE, band)
    os.makedirs(band_result_dir, exist_ok=True)
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_results = {}
    for clf_name, clf_dict in classifiers.items():
        print(f"\n  Training {clf_name}...")
        model = clf_dict["model"]
        param_grid = clf_dict["params"]
        acc_list, sens_list, spec_list, prec_list, f1_list, auc_list = [], [], [], [], [], []
        all_y_true, all_y_pred, all_y_proba = [], [], []
        best_params_list = []
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params_list.append(grid_search.best_params_)
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
            acc_list.append(accuracy_score(y_test, y_pred))
            sens_list.append(recall_score(y_test, y_pred, zero_division=0))
            spec_list.append(specificity_score_func(y_test, y_pred))
            prec_list.append(precision_score(y_test, y_pred, zero_division=0))
            f1_list.append(f1_score(y_test, y_pred, zero_division=0))
            if y_proba is not None:
                auc_list.append(roc_auc_score(y_test, y_proba))
            else:
                auc_list.append(np.nan)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            if y_proba is not None:
                all_y_proba.extend(y_proba)
        results = {
            "accuracy": f"{np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}",
            "sensitivity": f"{np.mean(sens_list):.4f} ± {np.std(sens_list):.4f}",
            "specificity": f"{np.mean(spec_list):.4f} ± {np.std(spec_list):.4f}",
            "precision": f"{np.mean(prec_list):.4f} ± {np.std(prec_list):.4f}",
            "f1_score": f"{np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}",
            "roc_auc": f"{np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}",
            "best_params": best_params_list
        }
        all_results[clf_name] = results
        report = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(os.path.join(band_result_dir, f"{clf_name}_classification_report.csv"))
        cm = confusion_matrix(all_y_true, all_y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy','MDD'], yticklabels=['Healthy','MDD'])
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.title(f'{clf_name} - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(band_result_dir, f"{clf_name}_confusion_matrix.png"))
        plt.close()
        if len(all_y_proba) > 0:
            fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
            auc_val = np.mean([a for a in auc_list if not np.isnan(a)]) if auc_list else 0
            plt.figure(figsize=(6,5))
            plt.plot(fpr, tpr, label=f'AUC = {auc_val:.3f}')
            plt.plot([0,1], [0,1], 'k--')
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title(f'{clf_name} - ROC Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(band_result_dir, f"{clf_name}_roc_curve.png"))
            plt.close()
        with open(os.path.join(band_result_dir, f"{clf_name}_best_params.json"), "w") as f:
            json.dump(best_params_list, f, indent=4)
        print(f"    AUC = {results['roc_auc']}")
    summary_df = pd.DataFrame(all_results).T
    summary_df.to_csv(os.path.join(band_result_dir, "model_comparison.csv"))
    print(f"\n  Saved all results for {band} in {band_result_dir}")
print("\n" + "="*50)
print(f"All classification tasks for {CONN_METRIC} completed.")
print(f"Results saved in: {RESULT_BASE}")