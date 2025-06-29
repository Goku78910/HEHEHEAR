import numpy as np
from sklearn.decomposition import PCA

# Example data
X = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9], 
              [10, 11, 12]])

# Initialize PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components

# Fit and transform the data
X_pca = pca.fit_transform(X)

# Print the original and transformed data
print("Original data:\n", X)
print("Transformed data:\n", X_pca)
print("Explained variance ratio:", pca.explained_variance_ratio_)
