import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = os.path.join(BASE_DIR, "images")
SCREE_PLOT_PATH = os.path.join(IMAGE_PATH, "scree_plot.png")
SCATTER_PLOT_PATH = os.path.join(IMAGE_PATH, "scatter_plot.png")
INSIGHTS_PATH = os.path.join(BASE_DIR, "insights")
PCA_FEATURES_PATH = os.path.join(INSIGHTS_PATH, "pca_features.txt")


np.random.seed(42)


def run_pca_exploration(X_train_scaled, y_train_enc):
    # 1. Fit PCA on the entire scaled training set
    # We don't limit n_components because we want to see the full variance
    pca = PCA()
    X_pca = pca.fit_transform(X_train_scaled)

    # ==========================================
    # TASK 1: THE SCREE PLOT (Explained Variance)
    # ==========================================
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(10, 5))
    plt.bar(
        range(1, len(explained_variance) + 1),
        explained_variance,
        alpha=0.5,
        align="center",
        label="Individual variance",
    )
    plt.step(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        where="mid",
        label="Cumulative variance",
        color="red",
    )

    # Draw a line at the 90% threshold
    plt.axhline(y=0.90, color="k", linestyle="--", label="90% Threshold")
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal component index")
    plt.title("PCA Scree Plot")
    plt.legend(loc="best")
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.tight_layout()
    plt.show()
    plt.savefig(SCREE_PLOT_PATH)

    # Find how many components are needed for 90%
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    print(f"Number of components needed to explain 90% of variance: {n_components_90}")

    # ==========================================
    # TASK 2: INSPECT PCA LOADINGS (Weights)
    # ==========================================
    # Create a DataFrame to see how much each original feature contributes to PC1 and PC2
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
        index=X_train_scaled.columns,
    )

    print("\n--- Top Features driving Principal Component 1 (PC1) ---")
    print(loadings["PC1"].abs().sort_values(ascending=False).head(5))

    print("\n--- Top Features driving Principal Component 2 (PC2) ---")
    print(loadings["PC2"].abs().sort_values(ascending=False).head(5))

    with open(PCA_FEATURES_PATH, "w") as f:
        f.write("--- Top Features driving Principal Component 1 (PC1) ---")
        f.write(str(loadings["PC1"].abs().sort_values(ascending=False).head(5)))
        f.write("\n--- Top Features driving Principal Component 2 (PC2) ---")
        f.write(str(loadings["PC2"].abs().sort_values(ascending=False).head(5)))

    # ==========================================
    # TASK 3: 2D SCATTER PLOT (Projected Data)
    # ==========================================
    plt.figure(figsize=(10, 8))

    # Scatter plot of PC1 vs PC2, colored by the target class
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_train_enc,
        cmap="viridis",
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    # Add a legend (Assuming 0=Black, 1=White, 2=Draw based on standard LabelEncoder)
    plt.legend(
        handles=scatter.legend_elements()[0], labels=["Class 0", "Class 1", "Class 2"]
    )
    plt.xlabel("Principal Component 1 (PC1)")
    plt.ylabel("Principal Component 2 (PC2)")
    plt.title("2D PCA Scatter Plot colored by Target Class")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(SCATTER_PLOT_PATH)
