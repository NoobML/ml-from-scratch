import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class KNN:
    """
    K-Nearest Neighbors classifier implemented from scratch.
    """

    def __init__(self, k=3):
        """
        Initialize KNN classifier.

        Args:
            k (int): Number of nearest neighbors to consider
        """
        self.k = k
        self.X_train = None  # Training features
        self.y_train = None  # Training labels

    def fit(self, X_train, y_train):
        """
        Store training data (lazy learning - no actual training).

        Args:
            X_train (np.ndarray): Training features, shape (n_samples, n_features)
            y_train (np.ndarray): Training labels, shape (n_samples,)
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict labels for test data using KNN algorithm.

        Args:
            X_test (np.ndarray): Test features, shape (n_test_samples, n_features)

        Returns:
            list: Predicted labels for test samples
        """
        # Step 1: Calculate Euclidean distances using vectorized operations
        # For efficiency, we use: ||a-b||² = ||a||² + ||b||² - 2(a·b)

        train_squared = np.sum(self.X_train ** 2, axis=1, keepdims=True)  # Shape: (n_train, 1)
        test_squared = np.sum(X_test ** 2, axis=1, keepdims=True)  # Shape: (n_test, 1)
        cross_term = X_test @ self.X_train.T  # Shape: (n_test, n_train)

        # Broadcasting: (n_test, 1) + (1, n_train) - 2*(n_test, n_train)
        squared_distances = test_squared + train_squared.T - 2 * cross_term
        distances = np.sqrt(squared_distances)  # Shape: (n_test, n_train)

        # Step 2: Find indices of k nearest neighbors for each test sample
        neighbor_indices = np.argsort(distances, axis=1)[:, :self.k]  # Shape: (n_test, k)

        # Step 3: Get labels of k nearest neighbors
        neighbor_labels = self.y_train[neighbor_indices]  # Shape: (n_test, k)

        # Step 4: Make predictions by majority voting
        predictions = []

        for test_idx in range(neighbor_labels.shape[0]):
            # Get k neighbor labels for current test sample
            current_neighbors = neighbor_labels[test_idx]  # Shape: (k,)

            # Count frequency of each label
            label_counts = {}
            for label in current_neighbors:
                label_counts[label] = label_counts.get(label, 0) + 1

            # Find label with highest frequency (majority vote)
            most_frequent_label = max(label_counts, key=label_counts.get)
            predictions.append(most_frequent_label)

        return predictions


def compare_knn_implementations():
    """
    Compare custom KNN implementation with sklearn's KNeighborsClassifier.
    Tests different k values and plots accuracy comparison.
    """
    # Load and prepare data
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Dataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print("-" * 50)

    # Test different k values
    k_values = range(1, 21)
    custom_accuracies = []
    sklearn_accuracies = []

    for k in k_values:
        # Custom KNN
        custom_knn = KNN(k=k)
        custom_knn.fit(X_train, y_train)
        custom_predictions = custom_knn.predict(X_test)
        custom_accuracy = np.mean(custom_predictions == y_test) * 100
        custom_accuracies.append(custom_accuracy)

        # Sklearn KNN
        sklearn_knn = KNeighborsClassifier(n_neighbors=k)
        sklearn_knn.fit(X_train, y_train)
        sklearn_predictions = sklearn_knn.predict(X_test)
        sklearn_accuracy = np.mean(sklearn_predictions == y_test) * 100
        sklearn_accuracies.append(sklearn_accuracy)

        print(f"k={k:2d} | Custom: {custom_accuracy:6.2f}% | Sklearn: {sklearn_accuracy:6.2f}%")

    # Plot comparison
    plt.figure(figsize=(12, 8))

    # Accuracy comparison plot
    plt.subplot(2, 1, 1)
    plt.plot(k_values, custom_accuracies, 'bo-', label='Custom KNN', linewidth=2, markersize=6)
    plt.plot(k_values, sklearn_accuracies, 'ro-', label='Sklearn KNN', linewidth=2, markersize=6)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy (%)')
    plt.title('KNN Accuracy Comparison: Custom vs Sklearn Implementation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Difference plot
    plt.subplot(2, 1, 2)
    accuracy_diff = np.array(custom_accuracies) - np.array(sklearn_accuracies)
    plt.plot(k_values, accuracy_diff, 'go-', linewidth=2, markersize=6)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy Difference (%)')
    plt.title('Accuracy Difference (Custom - Sklearn)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Best Custom KNN accuracy: {max(custom_accuracies):.2f}% (k={k_values[np.argmax(custom_accuracies)]})")
    print(f"Best Sklearn KNN accuracy: {max(sklearn_accuracies):.2f}% (k={k_values[np.argmax(sklearn_accuracies)]})")
    print(f"Average accuracy difference: {np.mean(accuracy_diff):.4f}%")
    print(f"Max accuracy difference: {np.max(np.abs(accuracy_diff)):.4f}%")

    if np.allclose(custom_accuracies, sklearn_accuracies, atol=1e-10):
        print("✅ Custom implementation matches sklearn perfectly!")
    else:
        print("⚠️  Minor differences detected (likely due to tie-breaking)")


if __name__ == "__main__":
    compare_knn_implementations()