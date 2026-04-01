# Entry point : runs the full pipeline

from src.pca_analysis import run_pca_exploration
from src.preprocessing import preprocess

if __name__ == "__main__":
    print("-" * 10)
    print("Start the preprocessing")
    X_train, X_validation, X_test, y_train, y_val, y_test = preprocess()
    print("End the preprocessing")
    print("-" * 10)

    # PCA analysis
    run_pca_exploration(X_train, y_train)
