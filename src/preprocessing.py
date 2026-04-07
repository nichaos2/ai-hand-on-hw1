# All preprocessing logic
import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, TargetEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, "models", "target_encoder.pkl")


np.random.seed(42)


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
    X_train, X_validation, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1111,
        random_state=42,
        stratify=y_temp,
    )

    return (X_train, X_validation, X_test, y_train, y_val, y_test)


def handle_missing_values(X_train, X_validation, X_test, y_train, y_val, y_test):
    """
    Handles missing values.
    """
    # 1. Target Column: Drop rows where target is missing
    train_mask = y_train.notna()
    X_train, y_train = X_train[train_mask].copy(), y_train[train_mask].copy()

    validation_mask = y_val.notna()
    X_validation, y_val = (
        X_validation[validation_mask].copy(),
        y_val[validation_mask].copy(),
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

    return (X_train, X_validation, X_test, y_train, y_val, y_test)


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
    target_encoder = TargetEncoder(
        smooth="auto", target_type="multiclass", cv=5, shuffle=True, random_state=42
    )
    # Fit on Train ONLY (using the encoded y_train)
    target_encoder.fit(X_train[high_cardinality_features], y_train_enc)

    # Get the new expanded column names generated by Scikit-Learn
    new_col_names = target_encoder.get_feature_names_out(high_cardinality_features)

    # Helper function to safely swap the columns
    def apply_target_encoding(X_df):
        X_df = X_df.copy()

        # Transform returns a numpy matrix of the 9 new columns
        encoded_data = target_encoder.transform(X_df[high_cardinality_features])

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

    # save target encoders
    joblib.dump(target_encoder, TARGET_ENCODER_PATH)

    return (
        X_train_encoded,
        X_validation_encoded,
        X_test_encoded,
        y_train_enc,
        y_val_enc,
        y_test_enc,
    )


def engineer_domain_features(X_train, X_validation, X_test):
    """
    Creates new domain-knowledge features and adds them alongside
    the original columns without dropping anything.
    """
    # for the api
    datasets = [X_train]

    # for training
    if isinstance(X_validation, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        datasets = [X_train, X_validation, X_test]

    for X in datasets:
        # --- Feature 1: Rating Advantage ---
        X["rating_advantage"] = X["white_rating"] - X["black_rating"]

        # --- Feature 2: Game Duration (Minutes) ---
        # Convert the massive Unix millisecond integers to datetime objects
        start_dt = pd.to_datetime(X["created_at"], unit="ms")
        end_dt = pd.to_datetime(X["last_move_at"], unit="ms")

        # Calculate the difference and convert to total minutes
        X["game_duration_mins"] = (end_dt - start_dt).dt.total_seconds() / 60

    return X_train, X_validation, X_test


def fix_zero_durations(X_train, X_validation, X_test):
    """
    Treats physically impossible 0-minute games as missing data
    and imputes them using the Training set's median duration.
    """
    # 1. Find the median duration in the Training set ONLY
    # We explicitly ignore the 0s so they don't drag the median down!
    valid_train_durations = X_train[X_train["game_duration_mins"] > 0][
        "game_duration_mins"
    ]
    median_duration = valid_train_durations.median()

    # 2. Replace the zeros with this median in all three sets
    for X in [X_train, X_validation, X_test]:
        # Wherever the duration is exactly 0, replace it with the median
        mask = X["game_duration_mins"] == 0
        X.loc[mask, "game_duration_mins"] = median_duration

    # print(f"Replaced 0-minute games with the Train median duration: {median_duration:.2f} mins")
    return X_train, X_validation, X_test


def fix_zero_durations_grouped(X_train, X_validation, X_test):
    """
    Imputes 0-minute games using the median duration of their specific time control.
    """
    # 1. Look ONLY at valid durations in the Training set
    valid_train = X_train[X_train["game_duration_mins"] > 0]

    # 2. Calculate the global median (as a safety fallback)
    global_median = valid_train["game_duration_mins"].median()

    # 3. Calculate the grouped medians
    # This creates a mapping like {"10+0": 14.5, "5+0": 6.2, "60+0": 58.1}
    grouped_medians = valid_train.groupby("increment_code")[
        "game_duration_mins"
    ].median()

    # 4. Apply the fix to all three datasets
    for X in [X_train, X_validation, X_test]:
        # Find exactly which rows have the 0s
        zero_mask = X["game_duration_mins"] == 0

        # Take the increment_code for those zero-rows, and map them to our grouped_medians
        # .fillna(global_median) handles any unknown formats in Val/Test!
        imputed_values = (
            X.loc[zero_mask, "increment_code"]
            .map(grouped_medians)
            .fillna(global_median)
        )

        # Inject the smart values back into the column
        X.loc[zero_mask, "game_duration_mins"] = imputed_values

    print("Zero-duration games successfully imputed using grouped time controls!")
    return X_train, X_validation, X_test


def scale_features_and_save(X_train, X_validation, X_test):
    """
    Fits a StandardScaler on the training numerical features,
    applies it to all sets, and saves the scaler to disk.
    """
    # 1. Define the continuous numerical columns that need scaling
    cols_to_scale = [
        "turns",
        "white_rating",
        "black_rating",
        "opening_ply",
        "created_at",
        "last_move_at",
        "rating_advantage",
        "game_duration_mins",
    ]

    # 2. Initialize the Scaler
    scaler = StandardScaler()

    # 3. FIT ON TRAINING SET ONLY
    # This learns the mean and variance from the training data
    scaler.fit(X_train[cols_to_scale])

    # 4. TRANSFORM ALL SETS
    # We use .copy() to avoid SettingWithCopyWarning in pandas
    X_train_scaled = X_train.copy()
    X_val_scaled = X_validation.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_val_scaled[cols_to_scale] = scaler.transform(X_validation[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # 5. SAVE THE FITTED SCALER FOR HOMEWORK 2
    with open(SCALER_PATH, "wb") as file:
        pickle.dump(scaler, file)

    print(f"Features scaled successfully. Scaler saved to {SCALER_PATH}")

    return X_train_scaled, X_val_scaled, X_test_scaled


def preprocess():

    df = load_data()
    (X_train, X_validation, X_test, y_train, y_val, y_test) = split_data(df=df)

    #
    (X_train, X_validation, X_test, y_train, y_val, y_test) = handle_missing_values(
        X_train, X_validation, X_test, y_train, y_val, y_test
    )

    #
    X_train, X_validation, X_test = detect_treat_outliers_iqr(
        X_train, X_validation, X_test
    )

    # we change the steps to engineer the domain features before the encoding in our case
    X_train, X_validation, X_test = engineer_domain_features(
        X_train, X_validation, X_test
    )

    # intermediate step to treat the "wrong" output for time control
    X_train, X_validation, X_test = fix_zero_durations_grouped(
        X_train, X_validation, X_test
    )

    X_train, X_validation, X_test, y_train, y_val, y_test = encode_categories(
        X_train, X_validation, X_test, y_train, y_val, y_test
    )

    X_train, X_validation, X_test = scale_features_and_save(
        X_train, X_validation, X_test
    )

    return X_train, X_validation, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    preprocess()
    print("Preprocessing DONE")

# def download_dataset():
#     import kagglehub
#     # TODO add logic if dataset exists
#     kagglehub.dataset_download(
#         "datasnaek/chess",
#         path=DATASET_PATH,
#     )

#     print("Path to dataset files:", DATASET_PATH)
