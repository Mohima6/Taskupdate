import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
from PIL import Image
BASE_PATH = r"C:\Users\mohimaCHAKRABORTY\Taskupdate\machine_learning\results\PLV\theta"
MODEL_COMPARISON_FILE = "model_comparison.csv"
CLASSIFIER_FILES = {
    "LogisticRegression": "LogisticRegression_classification_report.csv",
    "DecisionTree": "DecisionTree_classification_report.csv",
    "RandomForest": "RandomForest_classification_report.csv",
    "SVM": "SVM_classification_report.csv",
    "XGBoost": "XGBoost_classification_report.csv"
}
BEST_PARAMS_FILES = {
    "LogisticRegression": "LogisticRegression_best_params.json",
    "DecisionTree": "DecisionTree_best_params.json",
    "RandomForest": "RandomForest_best_params.json",
    "SVM": "SVM_best_params.json",
    "XGBoost": "XGBoost_best_params.json"
}
# Optional confusion matrix and ROC curve images
IMAGE_FILES = {
    "LogisticRegression": ["LogisticRegression_confusion_matrix.png", "LogisticRegression_roc_curve.png"],
    "DecisionTree": ["DecisionTree_confusion_matrix.png", "DecisionTree_roc_curve.png"],
    "RandomForest": ["RandomForest_confusion_matrix.png", "RandomForest_roc_curve.png"],
    "SVM": ["SVM_confusion_matrix.png", "SVM_roc_curve.png"],
    "XGBoost": ["XGBoost_confusion_matrix.png", "XGBoost_roc_curve.png"]
}
OUTPUT_PDF = os.path.join(BASE_PATH, "PLV_theta_classification_report.pdf")
def load_classification_report(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    return df
def load_best_params(json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)
    if params and len(params) > 0:
        first = params[0]
        return ", ".join([f"{k}={v}" for k, v in first.items()])
    return "N/A"
def dataframe_to_table(df, title, ax, fontsize=9):
    ax.axis('tight')
    ax.axis('off')
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df_display = df.copy()
    for col in df_display.columns:
        if pd.api.types.is_numeric_dtype(df_display[col]):
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "")
    data = df_display.values
    col_labels = df_display.columns.tolist()
    row_labels = df_display.index.tolist()
    table = ax.table(cellText=data,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.2, 1.5)
    ax.set_title(title, fontsize=12, weight='bold', y=1.05)
    return table
def embed_image(ax, img_path, title):
    try:
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=10)
        return True
    except Exception as e:
        ax.text(0.5, 0.5, f"Image not found:\n{img_path}", ha='center', va='center', fontsize=8)
        ax.axis('off')
        ax.set_title(title, fontsize=10)
        return False
print("Generating PDF report for PLV Theta band...")
print(f"Looking for files in: {BASE_PATH}")
if not os.path.exists(BASE_PATH):
    raise FileNotFoundError(f"Base path does not exist: {BASE_PATH}")
model_comp_path = os.path.join(BASE_PATH, MODEL_COMPARISON_FILE)
if not os.path.exists(model_comp_path):
    raise FileNotFoundError(f"Model comparison file not found: {model_comp_path}")
model_comp = pd.read_csv(model_comp_path, index_col=0)
model_comp.index.name = 'Classifier'
classifier_reports = {}
for clf, fname in CLASSIFIER_FILES.items():
    full_path = os.path.join(BASE_PATH, fname)
    if os.path.exists(full_path):
        classifier_reports[clf] = load_classification_report(full_path)
    else:
        print(f"Warning: {full_path} not found, skipping {clf}")
best_params = {}
for clf, fname in BEST_PARAMS_FILES.items():
    full_path = os.path.join(BASE_PATH, fname)
    if os.path.exists(full_path):
        best_params[clf] = load_best_params(full_path)
with PdfPages(OUTPUT_PDF) as pdf:
    fig, ax = plt.subplots(figsize=(12, 4))
    display_cols = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'roc_auc']
    existing_cols = [c for c in display_cols if c in model_comp.columns]
    summary_df = model_comp[existing_cols].copy()
    dataframe_to_table(summary_df, "Model Comparison (mean ± std)", ax, fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    for clf, df_report in classifier_reports.items():
        param_str = best_params.get(clf, "N/A")
        title = f"{clf} - Classification Report\nBest parameters (first fold): {param_str}"
        fig, ax = plt.subplots(figsize=(10, 4))
        rows_to_keep = ['0', '1', 'macro avg', 'weighted avg']
        existing_rows = [r for r in rows_to_keep if r in df_report.index]
        df_display = df_report.loc[existing_rows, ['precision', 'recall', 'f1-score', 'support']]
        dataframe_to_table(df_display, title, ax, fontsize=9)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        img_files = IMAGE_FILES.get(clf, [])
        if any(os.path.exists(os.path.join(BASE_PATH, img)) for img in img_files):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, img_name in zip(axes, img_files):
                img_path = os.path.join(BASE_PATH, img_name)
                embed_image(ax, img_path, img_name.replace('_', ' ').replace('.png', ''))
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    if best_params:
        fig, ax = plt.subplots(figsize=(12, 2 + 0.5 * len(best_params)))
        params_df = pd.DataFrame(list(best_params.items()), columns=['Classifier', 'Best Params (first fold)'])
        dataframe_to_table(params_df, "Best Hyperparameters (first outer fold)", ax, fontsize=9)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
print(f"PDF saved to: {OUTPUT_PDF}")
