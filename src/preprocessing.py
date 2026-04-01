# All preprocessing logic
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder

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
    y = df["winner"]
    X = df.drop(columns=["winner"])

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


def detect_treat_outliers_iqr(X_train, X_validation, X_test):
    """
    Learns IQR boundaries from the Training set only,
    and applies capping (Winsorization) to Train, Val, and Test sets.
    """
    # Define the continuous numerical columns to check
    # Note: We do NOT do this to categorical encoded data or IDs!
    features_to_cap = ["turns", "white_rating", "black_rating", "opening_ply"]

    # We will store the boundaries just in case we need to inspect them
    boundaries = {}

    for col in features_to_cap:
        # Compute boundaries on TRAIN only top prevent data leakage
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Save the rules for our records
        boundaries[col] = {"lower": lower_bound, "upper": upper_bound}

        # 2. TRANSFORM ALL SETS (Capping / Winsorizing)
        # np.clip forces values outside the range to equal the boundary exactly
        X_train[col] = np.clip(X_train[col], lower_bound, upper_bound)
        X_validation[col] = np.clip(X_validation[col], lower_bound, upper_bound)
        X_test[col] = np.clip(X_test[col], lower_bound, upper_bound)

    # print("Outliers successfully capped using Train-set IQR boundaries!")
    return X_train, X_validation, X_test


def encode_categories(X_train, X_validation, X_test, y_train, y_val, y_test):
    # One hot features: winner which is the target y
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    # Binary features: rated is True/False -> 1/0
    for X in [X_train, X_validation, X_test]:
        X["rated"] = X["rated"].astype(int)

    # Target encoding
    high_cardinality_features = ["increment_code", "opening_eco", "opening_name"]
    targer_encoder = TargetEncoder(
        smooth="auto", target_type="multiclass", cv=5, shuffle=True, random_state=42
    )
    # Fit on Train ONLY (using the encoded y_train)
    targer_encoder.fit(X_train[high_cardinality_features], y_train_enc)

    # Get the new expanded column names generated by Scikit-Learn
    new_col_names = targer_encoder.get_feature_names_out(high_cardinality_features)

    # Helper function to safely swap the columns
    def apply_target_encoding(X_df):
        X_df = X_df.copy()

        # Transform returns a numpy matrix of the 9 new columns
        encoded_data = targer_encoder.transform(X_df[high_cardinality_features])

        # Turn it into a DataFrame with the correct names and matching index
        encoded_df = pd.DataFrame(encoded_data, columns=new_col_names, index=X_df.index)

        # Drop the original text columns and merge the new probability columns
        X_df = X_df.drop(columns=high_cardinality_features)
        X_df = pd.concat([X_df, encoded_df], axis=1)

        return X_df

    # Transform all sets
    X_train_encoded = apply_target_encoding(X_train)
    X_validation_encoded = apply_target_encoding(X_validation)
    X_test_encoded = apply_target_encoding(X_test)

    return (
        X_train_encoded,
        X_validation_encoded,
        X_test_encoded,
        y_train_enc,
        y_val_enc,
        y_test_enc,
    )


# def download_dataset():
#     import kagglehub
#     # TODO add logic if dataset exists
#     kagglehub.dataset_download(
#         "datasnaek/chess",
#         path=DATASET_PATH,
#     )

#     print("Path to dataset files:", DATASET_PATH)
