# Entry point : runs the full pipeline

from src.preprocessing import (
    detect_treat_outliers_iqr,
    encode_categories,
    handle_missing_values,
    load_data,
    split_data,
)

if __name__ == "__main__":
    print("-" * 10)
    print("start the preprocessing")
    df = load_data()
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df=df)

    #
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = handle_missing_values(X_train, X_validation, X_test, y_train, y_val, y_test)

    #
    X_train, X_validation, X_test = detect_treat_outliers_iqr(
        X_train, X_validation, X_test
    )

    print(X_train.head())
    print(type(y_train))

    #
    # check_test_data(df, X_train, X_validation, X_test, y_train, y_val, y_test)

    X_train, X_validation, X_test, y_train, y_val, y_test = encode_categories(
        X_train, X_validation, X_test, y_train, y_val, y_test
    )
    print(X_train.head())
    print(type(y_train))
