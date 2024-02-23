import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Apply SelectKBest with chi2 to select all features for visualization
selector = SelectKBest(chi2, k='all')
selector.fit(X, y)

# Get scores of each feature
scores = selector.scores_

# Plotting
plt.bar(range(len(feature_names)), scores, color='skyblue')
plt.xticks(range(len(feature_names)), feature_names, rotation=45)
plt.xlabel('Features')
plt.ylabel('Chi2 Score')
plt.title('Feature Importance using Chi2 Test')
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()

# Identify selected features
selected_features = [feature_names[i] for i in range(len(selector.get_support())) if selector.get_support()[i]]

print("Selected Features:")
for feature in selected_features:
    print(feature)
