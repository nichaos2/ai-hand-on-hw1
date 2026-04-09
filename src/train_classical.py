# Classical ML training logic
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(BASE_DIR, "models", "classical_model.pkl")
INSIGHTS_PATH = os.path.join(BASE_DIR, "insights")
MODEL_REPORT_PATH = os.path.join(INSIGHTS_PATH, "classical_model_report.txt")
HYPERARAMETER_TUNING_PATH = os.path.join(INSIGHTS_PATH, "hyperparameter_tuning.txt")


def train_classical_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    tuning=False,
    X_test=None,
    y_test=None,
):
    # Initialize the XGBoost Classifier
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

    if tuning:
        _tune_hyperparameters(
            xgb_model, X_train, y_train, X_validation, y_validation, X_test, y_test
        )

    return xgb_model, importances


def _tune_hyperparameters(
    baseline_model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test=None,
    y_test=None,
):
    print("\n" + "=" * 50)
    print("STARTING HYPERPARAMETER TUNING")
    print("=" * 50)

    # GET BASELINE SCORES
    # We use the already-trained baseline model passed from the main function!
    base_val_probs = baseline_model.predict_proba(X_validation)
    base_val_auc = roc_auc_score(
        y_validation, base_val_probs, multi_class="ovr", average="weighted"
    )

    if X_test is not None and y_test is not None:
        base_test_probs = baseline_model.predict_proba(X_test)
        base_test_auc = roc_auc_score(
            y_test, base_test_probs, multi_class="ovr", average="weighted"
        )

    # DEFINE SEARCH SPACE
    param_distributions = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 6, 10, None],
        "min_child_weight": [1, 2, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    # RUN RANDOMIZED SEARCH (NO EARLY STOPPING)
    xgb_search_base = XGBClassifier(
        objective="multi:softprob", num_class=3, random_state=42
    )

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=xgb_search_base,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="roc_auc_ovr_weighted",
        cv=cv_strategy,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    print("Running RandomizedSearchCV (this might take a few minutes)...")
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_

    # TRAIN FINAL TUNED MODEL (WITH EARLY STOPPING)

    print("\nTraining final tuned model with early stopping on X_validation...")
    best_tuned_model = XGBClassifier(
        **best_params,  # Inject the winning parameters
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        early_stopping_rounds=20,  # Add early stopping
    )

    best_tuned_model.fit(
        X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False
    )

    # GET TUNED SCORES
    tuned_val_probs = best_tuned_model.predict_proba(X_validation)
    tuned_val_auc = roc_auc_score(
        y_validation, tuned_val_probs, multi_class="ovr", average="weighted"
    )

    if X_test is not None and y_test is not None:
        tuned_test_probs = best_tuned_model.predict_proba(X_test)
        tuned_test_auc = roc_auc_score(
            y_test, tuned_test_probs, multi_class="ovr", average="weighted"
        )

    # SAVE RESULTS FOR THE REPORT
    report_str = ""
    report_str += "HYPERPARAMETER TUNING RESULTS\n"
    report_str += "Search space\n"
    report_str += f"learning_rate: {param_distributions['learning_rate']}\n"
    report_str += f"n_estimators: {param_distributions['n_estimators']}\n"
    report_str += f"max_depth: {param_distributions['max_depth']}\n"
    report_str += f"min_child_weight: {param_distributions['min_child_weight']}\n"
    report_str += f"subsample: {param_distributions['subsample']}\n"
    report_str += f"colsample_bytree: {param_distributions['colsample_bytree']}\n"
    report_str += f"Result - Best params: {random_search.best_params_}\n"
    report_str += "-" * 40 + "\n"
    report_str += f"Validation confirmation - ROC-AUC on X_val: {base_val_auc:.4f} (baseline) -> {tuned_val_auc:.4f} (tuned)\n"
    report_str += f"Test set improvement    - ROC-AUC on X_test: {base_test_auc:.4f} (baseline) -> {tuned_test_auc:.4f} (tuned)"

    with open(HYPERARAMETER_TUNING_PATH, "w") as f:
        f.write(report_str)

    # Optional: Return the new model in case you want to save it later!
    # return best_tuned_model
