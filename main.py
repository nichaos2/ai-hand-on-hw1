# Entry point : runs the full pipeline

from src.pca_analysis import run_pca_exploration
from src.preprocessing import preprocess
from src.train_classical import train_classical_model

if __name__ == "__main__":
    print("-" * 10)
    print("Start the preprocessing")
    X_train, X_validation, X_test, y_train, y_validation, y_test = preprocess()
    print("End the preprocessing")
    print("-" * 10)

    # PCA analysis
    run_pca_exploration(X_train, y_train)

    # train classical model
    classical_model, feature_importances = train_classical_model(
        X_train, y_train, X_validation, y_validation
    )
