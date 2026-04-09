# Entry point : runs the full pipeline

from src.evaluate import evaluate_final_models
from src.pca_analysis import run_pca_exploration
from src.preprocessing import preprocess
from src.train_classical import train_classical_model
from src.train_neural import train_neural_network

# Default to False; set to True for Hyperparmeter Tuning of the XGBoost model
XGBOOST_TUNING = False

if __name__ == "__main__":
    print("START OD PIPELINE")
    print("-" * 10)
    print("Start the preprocessing")
    X_train, X_validation, X_test, y_train, y_validation, y_test = preprocess()
    print("End the preprocessing")
    print("-" * 10)

    # PCA analysis
    run_pca_exploration(X_train, y_train)

    print("-" * 10)
    print("Train the classical ML model")
    classical_model, feature_importances = train_classical_model(
        X_train, y_train, X_validation, y_validation, XGBOOST_TUNING, X_test, y_test
    )
    print("End training the classical ML model")
    print("-" * 10)

    print("-" * 10)
    print("Train the neural network")
    apply_plot = True  # True for experimentation
    nn_model = train_neural_network(
        X_train, y_train, X_validation, y_validation, apply_plot=apply_plot
    )
    print("End training the neural network")
    print("-" * 10)

    print("-" * 10)
    print("Evaluate and Compare models")
    evaluate_final_models(classical_model, nn_model, X_test, y_test)
    print("End of evaluation and comparison")
    print("-" * 10)

    print("END OF PIPELINE")
