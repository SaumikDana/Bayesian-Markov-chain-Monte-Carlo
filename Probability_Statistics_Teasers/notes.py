import matplotlib.pyplot as plt
import numpy as np

# Generate some random data points
x = np.random.rand(100)
y = 2 * x + 1 + np.random.randn(100)

# Fit a linear regression line
coefficients = np.polyfit(x, y, 1)
line = np.poly1d(coefficients)
xfit = np.linspace(0, 1, 100)
yfit = line(xfit)

# Plot the data points and the regression line
plt.scatter(x, y, label='Data Points')
plt.plot(xfit, yfit, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load the Iris dataset
data = load_iris()
X = data.data[:, 2:]  # Using only petal length and width
y = data.target

# Fit a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Plot the decision tree
fig, ax = plt.subplots(figsize=(8, 8))
tree.plot_tree(clf, filled=True, feature_names=data.feature_names[2:], class_names=data.target_names, ax=ax)
plt.show()

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
data = load_iris()
X = data.data[:, :2]  # Using only the first two features for simplicity
y = data.target

# Fit a random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# Create a meshgrid to plot the decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Make predictions on the meshgrid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Random Forest Decision Boundaries')
plt.show()