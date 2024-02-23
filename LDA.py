import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Apply PCA
pca = PCA(n_components=2)
pca.fit(X)
pca_components = pca.components_

# Calculating squared loadings for PCA feature importance
pca_feature_importance = pd.DataFrame(pca_components**2, columns=feature_names, index=['PC1', 'PC2'])

# Apply LDA
lda = LDA(n_components=2)
lda.fit(X, y)
lda_coefs = lda.scalings_

# Convert LDA coefficients to DataFrame for easier plotting
lda_feature_contributions = pd.DataFrame(lda_coefs, columns=['LD1', 'LD2'], index=feature_names)

# Plotting both PCA feature importance and LDA feature contributions
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# PCA Feature Importance
pca_feature_importance.T.plot(kind='bar', ax=axes[0], colormap='viridis')
axes[0].set_title('PCA Feature Importance')
axes[0].set_ylabel('Squared Loadings')
axes[0].set_xlabel('Features')
axes[0].legend(title='PCA Components')

# LDA Feature Contributions
lda_feature_contributions.plot(kind='bar', ax=axes[1], colormap='viridis')
axes[1].set_title('LDA Feature Contributions')
axes[1].set_ylabel('Coefficients')
axes[1].set_xlabel('Features')
axes[1].legend(title='LDA Components')

plt.tight_layout()
plt.show()
