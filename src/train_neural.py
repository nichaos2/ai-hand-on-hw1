# Neural network training logic
import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(BASE_DIR, "models", "neural_network.pt")
INSIGHTS_PATH = os.path.join(BASE_DIR, "insights")
MODEL_REPORT_PATH = os.path.join(INSIGHTS_PATH, "neural_network_report.txt")
IMAGE_PATH = os.path.join(BASE_DIR, "images")
LOSS_CURVES_PATH = os.path.join(IMAGE_PATH, "loss_curves.png")

DROPOUT = 0.2


# ==========================================
# DEFINE THE ARCHITECTURE
# ==========================================
class ChessNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ChessNN, self).__init__()

        # Hidden Layer 1 (Input -> 64 nodes)
        self.fc1 = nn.Linear(input_size, 64)

        # ACTIVATION
        self.act1 = nn.ReLU()
        # EXPERIMENT: uncomment the following line to test with tanh
        # self.act1 = nn.Tanh()
        # Dropout for Regularization (0.2 = 20% of nodes randomly turned off)
        self.drop1 = nn.Dropout(p=DROPOUT)

        # Hidden Layer 2 (64 nodes -> 32 nodes)
        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()
        # EXPERIMENT: uncomment the following line to test with tanh
        # self.act2 = nn.Tanh()
        self.drop2 = nn.Dropout(p=DROPOUT)

        # Output Layer (32 nodes -> 3 classes)
        # We leave this as linear (no activation) because CrossEntropyLoss applies Softmax!
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)

        x = self.out(x)
        return x


# ==========================================
# CONVERT DATA TO PYTORCH TENSORS
# ==========================================
# Neural networks require float32 for features and long (integers) for targets
def convert_data_to_pytorch_tensors(X_train, y_train, X_validation, y_validation):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_val_tensor = torch.tensor(X_validation.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_validation, dtype=torch.long)

    # Create DataLoaders for batching (batch size of 64 is a good default)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    return X_train_tensor, train_dataset, train_loader, val_dataset, val_loader


def train_model(
    train_dataset,
    train_loader,
    val_dataset,
    val_loader,
    model,
    apply_plot: bool = False,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 150
    patience = 25  # How many epochs to wait before stopping if no improvement
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_weights = None

    train_losses = []
    val_losses = []

    print("Starting Neural Network Training...")

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()  # Clear old gradients
            predictions = model(batch_X)  # Forward pass
            loss = criterion(predictions, batch_y)  # Calculate Loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_train_loss += loss.item() * batch_X.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # --- VALIDATION PHASE ---
        model.eval()  # Turn off Dropout for validation!
        running_val_loss = 0.0

        with torch.no_grad():  # Don't calculate gradients to save memory
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                running_val_loss += loss.item() * batch_X.size(0)

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        # --- EARLY STOPPING CHECK ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Deepcopy saves the exact weights from this specific "best" epoch
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}"
            )

        if epochs_no_improve >= patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch}! Reverting to best weights from {epoch - patience}."
            )
            model.load_state_dict(best_model_weights)
            break

    # Ensure we load the best weights if training finished without early stopping
    if epochs_no_improve < patience:
        model.load_state_dict(best_model_weights)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")

    if apply_plot:
        plot_loss_functions(train_losses, val_losses)

    return model


def plot_loss_functions(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.title("Neural Network Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.savefig(LOSS_CURVES_PATH)


def train_neural_network(
    X_train, y_train, X_validation, y_validation, apply_plot: bool = False
):
    X_train_tensor, train_dataset, train_loader, val_dataset, val_loader = (
        convert_data_to_pytorch_tensors(X_train, y_train, X_validation, y_validation)
    )
    input_dim = X_train_tensor.shape[1]  # Should be 18 based on our previous steps
    model = ChessNN(input_size=input_dim, num_classes=3)
    model = train_model(
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        model,
        apply_plot=apply_plot,
    )
    return model
