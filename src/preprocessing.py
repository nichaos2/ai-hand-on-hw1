# All preprocessing logic
import os
from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
# DATASET_PATH = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")


def load_data() -> pd.DataFrame:
    """Loads the CSV and performs initial 'raw' target conversion."""
    path = Path(DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"No file found at {path}")

    df = pd.read_csv(path)

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


def download_dataset():
    # TODO add logic if dataset exists
    kagglehub.dataset_download(
        "datasnaek/chess",
        path=DATASET_PATH,
    )

    print("Path to dataset files:", DATASET_PATH)
