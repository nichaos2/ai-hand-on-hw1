# Entry point : runs the full pipeline

from src.preprocessing import preprocess
from src.train_neural import train_neural_network

if __name__ == "__main__":
    print("-" * 10)
    print("Start the preprocessing")
    X_train, X_validation, X_test, y_train, y_validation, y_test = preprocess()
    print("End the preprocessing")
    print("-" * 10)

    # PCA analysis
    # run_pca_exploration(X_train, y_train)

    # train classical model
    # classical_model, feature_importances = train_classical_model(
    #     X_train, y_train, X_validation, y_validation
    # )

    train_neural_network(X_train, y_train, X_validation, y_validation)
