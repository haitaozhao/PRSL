import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, _ = X.shape
        weights = np.full(n_samples, (1 / n_samples))
        y_encoded = np.where(y == 0, -1, 1)  # Encode labels as -1 or 1
        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y_encoded, sample_weight=weights)
            y_pred = model.predict(X)
            err = np.sum(weights * (y_pred != y_encoded))
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            weights *= np.exp(-alpha * y_encoded * y_pred)
            weights /= np.sum(weights)
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        return np.sign(np.dot(self.alphas, preds))

def plot_decision_boundary(X, y, models, alphas, iteration):
    plt.figure(figsize=(8, 6))
    h = 0.02  # step size in the mesh

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.subplot(2, 3, iteration)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = np.array([model.predict(np.c_[xx.ravel(), yy.ravel()]) for model in models[:iteration]])
    Z = np.dot(alphas[:iteration], Z)

    Z = np.sign(Z)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'Iteration {iteration}')

    plt.show()

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Keep only the last two classes (versicolor and virginica)
X = X[y != 0, :2]
y = y[y != 0]-1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Adaboost classifier
adaboost = AdaBoost(n_estimators=5)
adaboost.fit(X_train, y_train)

# Plot the decision boundaries for each iteration
for i in range(1, 6):
    plot_decision_boundary(X_train, y_train, adaboost.models, adaboost.alphas, i)
