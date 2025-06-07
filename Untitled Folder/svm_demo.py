import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Selecting only the first two features
y = iris.target

# Keep only the samples corresponding to the first two classes
X = X[y != 2]
y = y[y != 2]

# SVM classifier
svm = SVC(kernel='linear', C=100)
svm.fit(X, y)

# Plot decision boundary and support vectors
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
# Plot support vectors
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k', label='Support Vectors')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM classification of Iris Setosa and Versicolor with Support Vectors')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Taking only the first two features
y = iris.target

# Select only the samples corresponding to Setosa and Versicolor classes
X = X[y != 2]
y = y[y != 2]

# Fit SVM model
model = SVC(kernel='linear')
model.fit(X, y)

# Get support vectors
support_vectors = model.support_vectors_

# Plotting the decision boundary and support vectors
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.show()
