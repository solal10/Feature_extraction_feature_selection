import numpy as np  # Add this import statement
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load dataset
train_df = pd.read_csv('train.csv')

# Select features and target variable
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Define categorical and numerical features
categorical_features = ['Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']

# Preprocessing for numerical features: fill missing values with median and standardize
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features: fill missing values with most frequent value and encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Apply LDA for dimensionality reduction
n_classes = len(pd.unique(y))
lda = LDA(n_components=min(n_classes - 1, 2))
X_lda = lda.fit_transform(X_preprocessed.toarray(), y)

# Get feature names after one-hot encoding for categorical features
one_hot_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)

# Initialize a dictionary to hold the summed absolute weights for each original categorical feature
cat_feature_weight_sums = dict.fromkeys(categorical_features, 0)

# Sum the absolute values of the weights for each one-hot encoded feature back to its original categorical feature
for original_feature in categorical_features:
    # Filter the one-hot encoded features for the current original categorical feature
    encoded_feature_indices = [i for i, feature in enumerate(one_hot_feature_names) if feature.startswith(original_feature)]
    # Sum the absolute weights for these features from lda.coef_
    cat_feature_weight_sums[original_feature] = sum(abs(lda.coef_[0, len(numerical_features) + np.array(encoded_feature_indices)]))

# Combine the weights for numerical and aggregated categorical features
feature_weights = list(lda.coef_[0, :len(numerical_features)]) + list(cat_feature_weight_sums.values())
feature_names = numerical_features + list(cat_feature_weight_sums.keys())

# Create a DataFrame for visualization
coef_df = pd.DataFrame(feature_weights, index=feature_names, columns=['Feature Importance'])
coef_df.plot(kind='bar', figsize=(12, 6))
plt.title('Feature Importance in LDA')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
