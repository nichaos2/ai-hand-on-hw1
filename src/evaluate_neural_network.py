import torch
from sklearn.metrics import accuracy_score, classification_report


def evaluate_neural_network(model, val_loader):
    # 1. Put the model in evaluation mode (turns off Dropout)
    model.eval()

    all_predictions = []
    all_true_labels = []

    # 2. Run the validation set through the model without tracking gradients
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            # Get raw mathematical outputs (logits)
            raw_outputs = model(batch_X)

            # Find the class (0, 1, or 2) with the highest probability score
            _, predicted_classes = torch.max(raw_outputs, 1)

            # Store the predictions and true labels
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_true_labels.extend(batch_y.cpu().numpy())

    # 3. Print the identical report we used for XGBoost
    val_accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"Neural Network Validation Accuracy: {val_accuracy:.4f}\n")
    print("Neural Network Classification Report:\n")
    print(classification_report(all_true_labels, all_predictions, zero_division=0))


# --- EXECUTION ---
# Make sure to pass in the val_loader we created earlier!
evaluate_neural_network(model, val_loader)
