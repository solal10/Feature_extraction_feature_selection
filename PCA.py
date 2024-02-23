import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Apply PCA and fit to the data
pca = PCA(n_components=2)
pca.fit(X)
# Calculate squared loadings for PCA components
pca_components_squared = pd.DataFrame(pca.components_**2, columns=feature_names, index=['PC1', 'PC2'])

# Plotting PCA Feature Importance
plt.figure(figsize=(10, 6))
pca_components_squared.T.plot(kind='bar')
plt.title('PCA Feature Importance')
plt.ylabel('Squared Loadings')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.legend(title='PCA Components')
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()
