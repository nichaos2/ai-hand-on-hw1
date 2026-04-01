# Entry point : runs the full pipeline

from src.preprocessing import preprocess

if __name__ == "__main__":
    print("-" * 10)
    print("Start the preprocessing")
    X_train, X_validation, X_test, y_train, y_val, y_test = preprocess()
    print("End the preprocessing")
    print("-" * 10)
