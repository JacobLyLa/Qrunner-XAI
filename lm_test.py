import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

# Generate artificial data
np.random.seed(0)
n_samples = 100
X = np.random.rand(n_samples, 3)
y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.5).astype(int)
# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # standardize makes it work...
print(X[:10])
print(y[:10])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SGDClassifier
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Get the coefficients of the decision boundary
coefficients = clf.coef_[0]
print("Coefficients:", coefficients)
print("Intercept:", clf.intercept_[0]) # Doesn't matter for orthogonal vector

# Normalize the coefficients
norm = np.linalg.norm(coefficients)
normalized_coefficients = - coefficients / norm
print("Normalized coefficients:", normalized_coefficients)

# Negate the coefficients to get the orthogonal vector towards class 0
orthogonal_vector = -normalized_coefficients

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# Plot points from class 0 and class 1
class_0_indices = np.where(y_train == 0)[0]
class_1_indices = np.where(y_train == 1)[0]
ax.scatter(X_train[class_0_indices, 0], X_train[class_0_indices, 1], X_train[class_0_indices, 2], c='b', marker='o', label='Class 0')
ax.scatter(X_train[class_1_indices, 0], X_train[class_1_indices, 1], X_train[class_1_indices, 2], c='r', marker='x', label='Class 1')

# Use cosine similarity on test samples and orthogonal vector
cosine_similarities = np.dot(X_test, orthogonal_vector)/(np.linalg.norm(X_test, axis=1) * np.linalg.norm(orthogonal_vector))
print("Cosine similarities:", cosine_similarities)
# Plot test points and color them according to the cosine similarity
test_labels = np.where(cosine_similarities > 0, 1, 0)
test_class_0_indices = np.where(test_labels == 0)[0]
test_class_1_indices = np.where(test_labels == 1)[0]
# Plot in different markers
ax.scatter(X_test[test_class_0_indices, 0], X_test[test_class_0_indices, 1], X_test[test_class_0_indices, 2], c='b', marker='s', label='Class 0 (test)')
ax.scatter(X_test[test_class_1_indices, 0], X_test[test_class_1_indices, 1], X_test[test_class_1_indices, 2], c='r', marker='^', label='Class 1 (test)')

# Create a meshgrid to plot the decision boundary
xx, yy = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
zz = (-clf.intercept_[0] - clf.coef_[0][0] * xx - clf.coef_[0][1] * yy) / clf.coef_[0][2]
ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)

# Plot the orthogonal vector
point_on_boundary = [0,0,0]
ax.quiver(point_on_boundary[0], point_on_boundary[1], point_on_boundary[2],
          orthogonal_vector[0], orthogonal_vector[1], orthogonal_vector[2],
          color='green', label='Orthogonal Vector')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

plt.legend()
plt.show()