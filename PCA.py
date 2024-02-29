import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load your dataset
train_data = pd.read_csv('train.csv')

# Preprocess the dataset
# Select features - dropping non-numeric columns for simplicity. Adjust as needed.
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X = train_data[features]

# Handling missing values - simple strategy for demonstration
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Target variable
y = train_data['Survived']

# No direct equivalent to feature_names in this case, creating custom names
feature_names = features

# Apply PCA and fit to the data
pca = PCA(n_components=2)
pca.fit(X_scaled)
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
