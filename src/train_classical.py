# Classical ML training logic
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(BASE_DIR, "models", "classical_model.pkl")
INSIGHTS_PATH = os.path.join(BASE_DIR, "insights")
MODEL_REPORT_PATH = os.path.join(INSIGHTS_PATH, "classical_model_report.txt")


def train_classical_model(X_train, y_train, X_validation, y_validation):
    #  Initialize the XGBoost Classifier
    # We set early_stopping_rounds here (standard in modern XGBoost versions)
    xgb_model = XGBClassifier(
        n_estimators=500,  # Maximum number of trees to build
        max_depth=6,  # Depth of each tree
        learning_rate=0.1,
        objective="multi:softprob",  # Multiclass classification
        num_class=3,  # 0 (Black), 1 (White), 2 (Draw)
        random_state=42,
        early_stopping_rounds=20,  # Stop if validation score doesn't improve for 20 rounds
    )

    print("Training XGBoost with early stopping...")

    # Fit the model using the validation set for monitoring
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_validation, y_validation)],
        verbose=50,  # Print progress every 50 trees
    )

    # Evaluate Validation Performance
    y_validation_pred = xgb_model.predict(X_validation)
    val_accuracy = accuracy_score(y_validation, y_validation_pred)
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    print(
        "\nClassification Report:\n",
        classification_report(y_validation, y_validation_pred),
    )

    # Extract Feature Importances
    importances = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": xgb_model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    print("\n--- Top 5 Feature Importances ---")
    print(importances.head(5))

    with open(MODEL_REPORT_PATH, "w") as f:
        f.write(f"\nValidation Accuracy: {val_accuracy:.4f}")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_validation, y_validation_pred))
        f.write("--- Top Features driving Principal Component 1 (PC1) ---")
        f.write(str(importances.head(5)))

    # Save the model to a pickle file
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"\nModel successfully saved to {MODEL_PATH}")

    return xgb_model, importances
