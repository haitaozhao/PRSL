import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.utils import resample

# 创建样本数据
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, random_state=42)

# 初始化弱分类器
weak_classifier = DecisionTreeClassifier(max_depth=1)

# 初始化AdaBoost
adaboost = AdaBoostClassifier(base_estimator=weak_classifier, n_estimators=50)

# 训练AdaBoost
adaboost.fit(X, y)

# 绘制初始分类面
def plot_decision_boundary(X, y, classifier, ax, title):
    h = .02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    ax.set_title(title)
    return scatter

# 获取每一轮的分类器及其样本权重
plt.figure(figsize=(15, 10))
for i in range(1, 6):  # 展示前5轮
    # 通过调整权重进行重采样
    sample_weight = adaboost.estimator_weights_[:i]
    classifier = adaboost.estimators_[i-1]
    
    ax = plt.subplot(2, 3, i)
    plot_decision_boundary(X, y, classifier, ax, f"Iteration {i}")
    
    # 在图中添加样本的权重
    ax.scatter(X[:, 0], X[:, 1], c='black', s=50, alpha=sample_weight[-1], edgecolors='r')

plt.tight_layout()
plt.show()
