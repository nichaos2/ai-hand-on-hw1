def check_test_data(df, X_train, X_val, X_test, y_train, y_val, y_test):
    # Verification
    print(f"Training set:   {len(X_train)} rows ({len(X_train) / len(df):.0%})")
    print(f"Validation set: {len(X_val)} rows ({len(X_val) / len(df):.0%})")
    print(f"Test set:       {len(X_test)} rows ({len(X_test) / len(df):.0%})")

    # Check stratification (Draw percentage)
    print("\nDraw percentage in each set:")
    print(f"Train: {(y_train == 'draw').mean():.2%}")
    print(f"Val:   {(y_val == 'draw').mean():.2%}")
    print(f"Test:  {(y_test == 'draw').mean():.2%}")
    print("-" * 10)
