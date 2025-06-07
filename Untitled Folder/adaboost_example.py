# code of adaboost to classify the last two classes of the iris data. 
# plot the decision boundary obtained in each iteration and show the sample size according to the weights in each iteration.

# show the decsion boundaries in subplots in two row in one figure.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

iris = load_iris()
XX = iris.data[:, 2:]  # using only the last two features for visualization
y = iris.target
XX = XX[y >= 1]  # Selecting only the last two classes
y = y[y >= 1]
y[y == 1] = -1  # Convert class 1 to -1 for AdaBoost
scaler = StandardScaler()
X = scaler.fit_transform(XX)
def adaboost(X, y, num_estimators):
    n_samples = X.shape[0]
    weights = np.ones(n_samples) / n_samples  # initialize weights

    models = []  # to store the weak classifiers
    alphas = []  # to store the corresponding alpha values

    num_cols = num_estimators // 2
    fig, axes = plt.subplots(nrows=2, ncols=num_cols, figsize=(15, 6))

    for t in range(num_estimators):
        # Train a weak classifier
        model = DecisionTreeClassifier(max_depth=1)
        model.fit(X, y, sample_weight=weights)

        # Predictions
        y_pred = model.predict(X)

        # Calculate weighted error
        err = np.sum(weights * (y != y_pred))

        # Calculate alpha (classifier weight)
        alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))

        # Update sample weights
        weights *= np.exp(-alpha * y * y_pred)
        weights /= np.sum(weights)

        # Store the model and its corresponding alpha
        models.append(model)
        alphas.append(alpha)

        # Plot decision boundary
        row = t // num_cols
        col = t % num_cols
        axes[row, col].set_title(f"Iteration {t+1}")
        plot_decision_boundary(X, y, model, ax=axes[row, col])
        axes[row, col].scatter(X[:, 0], X[:, 1], s=weights * 6000, c='red', alpha=0.5)  # Sample size according to weights

    plt.tight_layout()
    plt.show()

    return models, alphas

def plot_decision_boundary(X, y, model, ax=None):
    if ax is None:
        ax = plt.gca()

    # Plotting decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.plasma, s=50, edgecolors='g')
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())




num_estimators = 8 # Number of weak classifiers (decision stumps)
models, alphas = adaboost(X, y, num_estimators)

def combined_decision_boundary(X, models, alphas):
    n_samples = X.shape[0]
    fx = np.zeros(n_samples)

    for model, alpha in zip(models, alphas):
        fx += alpha * model.predict(X)

    # Combine the decisions of all weak classifiers weighted by their corresponding alpha values
    final_pred = np.sign(fx)

    return final_pred

def plot_decision_boundary(X, y, models, alphas):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = combined_decision_boundary(np.c_[xx.ravel(), yy.ravel()], models, alphas)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=60, edgecolors='k')
    plt.title("Combined Decision Boundary")
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

# Call the function to plot the combined decision boundary
plot_decision_boundary(X, y, models, alphas)
