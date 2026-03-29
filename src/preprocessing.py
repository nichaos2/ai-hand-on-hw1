# All preprocessing logic
import os
from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")


def load_data() -> pd.DataFrame:
    """Loads the CSV and performs initial 'raw' target conversion."""
    path = Path(DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"No file found at {path}")

    df = pd.read_csv(path)

    # .str.split() with expand=True splits the forrmat "10+5" into two separate columns
    # time_split = df["increment_code"].str.split("+", expand=True)

    # # Convert them safely to numbers (errors='coerce' turns weird data into NaNs)
    # df["base_time_mins"] = pd.to_numeric(time_split[0], errors="coerce")
    # df["increment_secs"] = pd.to_numeric(time_split[1], errors="coerce")

    # The columns we know we NEVER want in our models
    cols_to_drop = [
        "id",  # unique identifier - does not train the model
        "white_id",  # High cardinality username
        "black_id",  # High cardinality username
        "victory_status",  # Target leakage
        "moves",  # Too complex for this iteration
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


def split_data(df):
    # 1. Define our Target (y) and Features (X)
    # For now, we keep all columns; we will select specific features in Step 2
    X = df.drop(columns=["winner"])
    y = df["winner"]

    # 2. First Split: Hold out the 10% Test Set
    # 'stratify=y' ensures the 5% draw ratio is preserved in the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=42,
        stratify=y,
    )

    # 3. Second Split: Separate Training (80% total) and Validation (10% total)
    # Since X_temp is 90% of the original, taking ~11.1% of it gives us 10% of the total
    # (0.111 * 0.90 ≈ 0.10)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1111,
        random_state=42,
        stratify=y_temp,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


def handle_missing_values(X_train, X_validation, X_test, y_train, y_validation, y_test):
    """
    Handles missing values.
    """
    # 1. Target Column: Drop rows where target is missing
    train_mask = y_train.notna()
    X_train, y_train = X_train[train_mask].copy(), y_train[train_mask].copy()

    validation_mask = y_validation.notna()
    X_validation, y_validation = (
        X_validation[validation_mask].copy(),
        y_validation[validation_mask].copy(),
    )

    test_mask = y_test.notna()
    X_test, y_test = X_test[test_mask].copy(), y_test[test_mask].copy()

    # Define our column types
    num_cols = [
        "turns",
        "white_rating",
        "black_rating",
        "opening_ply",
        "created_at",
        "last_move_at",
        # "base_time_mins",
        # "increment_secs",
    ]
    cat_cols = [col for col in X_train.columns if col not in num_cols]
    # Force mixed types into pure Python objects to avoid error for the column opening_eco
    # This prevents Scikit-Learn from attempting a 'float' conversion.
    for df in [X_train, X_validation, X_test]:
        df[cat_cols] = df[cat_cols].astype("object")

    # Numerical Imputation (Median)
    num_imputer = SimpleImputer(strategy="median")

    # FIT ONLY ON TRAINING SET
    num_imputer.fit(X_train[num_cols])

    # TRANSFORM ON ALL SETS
    X_train.loc[:, num_cols] = num_imputer.transform(X_train[num_cols])
    X_validation.loc[:, num_cols] = num_imputer.transform(X_validation[num_cols])
    X_test.loc[:, num_cols] = num_imputer.transform(X_test[num_cols])

    # Categorical Imputation (Mode / Most Frequent)
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # FIT ONLY ON TRAINING SET
    cat_imputer.fit(X_train[cat_cols])

    # TRANSFORM ALL SETS
    X_train.loc[:, cat_cols] = cat_imputer.transform(X_train[cat_cols])
    X_validation.loc[:, cat_cols] = cat_imputer.transform(X_validation[cat_cols])
    X_test.loc[:, cat_cols] = cat_imputer.transform(X_test[cat_cols])

    return (X_train, X_validation, X_test, y_train, y_validation, y_test)


def download_dataset():
    # TODO add logic if dataset exists
    kagglehub.dataset_download(
        "datasnaek/chess",
        path=DATASET_PATH,
    )

    print("Path to dataset files:", DATASET_PATH)
