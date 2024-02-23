import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
train_data = pd.read_csv('train.csv')

# Define features and target variable
X = train_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
y = train_data['Survived']

# Specify numerical and categorical features
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Sex', 'Embarked']

# Define transformers for numerical and categorical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model with Lasso within a pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', Lasso(alpha=0.1))])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model
model.fit(X_train, y_train)


# Function to extract feature names after preprocessing
def get_feature_names(column_transformer):
    output_features = []

    # Process categorical features to get feature names
    for transformer in column_transformer.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
            categorical_features):
        output_features.append(transformer)

    # Add numerical features as is (assuming they're passed through without changes)
    output_features.extend(numerical_features)

    return output_features


# Extract the coefficients
coefficients = model.named_steps['classifier'].coef_

# Get the feature names from the preprocessor
feature_names = get_feature_names(preprocessor)

# Plot the Lasso coefficients
plt.figure(figsize=(10, 6))
plt.bar(range(len(coefficients)), coefficients, tick_label=feature_names)
plt.xticks(rotation=45, ha="right")
plt.ylabel('Coefficient Value')
plt.title('Feature Importance (Lasso Coefficients)')
plt.tight_layout()
plt.show()
