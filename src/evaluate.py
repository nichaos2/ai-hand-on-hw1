# Evaluation and visualization
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
INSIGHTS_PATH = os.path.join(BASE_DIR, "insights")
MODEL_COMPARISON_PATH = os.path.join(INSIGHTS_PATH, "model_comparison.txt")
IMAGE_PATH = os.path.join(BASE_DIR, "images")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(IMAGE_PATH, "confusion_matrix.png")

torch.manual_seed(42)


def evaluate_final_models(xgb_model, nn_model, X_test_scaled, y_test_enc):
    # ==========================================
    # XGBOOST EVALUATION
    # ==========================================
    xgb_preds = xgb_model.predict(X_test_scaled)
    xgb_probs = xgb_model.predict_proba(X_test_scaled)

    xgb_acc = accuracy_score(y_test_enc, xgb_preds)
    xgb_prec, xgb_rec, xgb_f1, _ = precision_recall_fscore_support(
        y_test_enc, xgb_preds, average="weighted", zero_division=0
    )
    xgb_roc = roc_auc_score(
        y_test_enc, xgb_probs, multi_class="ovr", average="weighted"
    )

    # ==========================================
    # NEURAL NETWORK EVALUATION
    # ==========================================
    # Convert test features to tensors (No need for y_test tensor!)
    X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)

    nn_model.eval()  # Turn off dropout!
    with torch.no_grad():
        nn_logits = nn_model(X_test_tensor)
        nn_probs = F.softmax(nn_logits, dim=1).numpy()
        # Get the predicted class (0, 1, or 2) and convert to numpy array
        nn_preds = torch.argmax(nn_logits, dim=1).numpy()

    # Evaluate NN predictions against the standard y_test_enc
    nn_acc = accuracy_score(y_test_enc, nn_preds)
    nn_prec, nn_rec, nn_f1, _ = precision_recall_fscore_support(
        y_test_enc, nn_preds, average="weighted", zero_division=0
    )
    nn_roc = roc_auc_score(y_test_enc, nn_probs, multi_class="ovr", average="weighted")

    # ==========================================
    # PRINT SIDE-BY-SIDE TABLE
    # ==========================================
    comparison_str = ""
    comparison_str += "Metric\t\tXGBoost\t\tNeural Network\n"
    comparison_str += f"Accuracy\t{xgb_acc:.4f}\t\t{nn_acc:.4f}\n"
    comparison_str += f"Precision\t{xgb_prec:.4f}\t\t{nn_prec:.4f}\n"
    comparison_str += f"Recall\t\t{xgb_rec:.4f}\t\t{nn_rec:.4f}\n"
    comparison_str += f"F1-score\t{xgb_f1:.4f}\t\t{nn_f1:.4f}\n"
    comparison_str += f"ROC-AUC\t\t{xgb_roc:.4f}\t\t{nn_roc:.4f}\n"
    print(comparison_str)
    with open(MODEL_COMPARISON_PATH, "w") as f:
        f.write(comparison_str)

    # ==========================================
    # PLOT CONFUSION MATRICES
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_xgb = confusion_matrix(y_test_enc, xgb_preds)
    disp_xgb = ConfusionMatrixDisplay(
        confusion_matrix=cm_xgb, display_labels=["Black", "Draw", "White"]
    )
    disp_xgb.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("XGBoost Confusion Matrix")

    cm_nn = confusion_matrix(y_test_enc, nn_preds)
    disp_nn = ConfusionMatrixDisplay(
        confusion_matrix=cm_nn, display_labels=["Black", "Draw", "White"]
    )
    disp_nn.plot(ax=axes[1], cmap="Reds", colorbar=False)
    axes[1].set_title("Neural Network Confusion Matrix")

    plt.tight_layout()
    plt.show()
    plt.savefig(CONFUSION_MATRIX_PLOT_PATH)
