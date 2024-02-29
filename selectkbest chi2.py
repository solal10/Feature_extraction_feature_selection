import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Assuming train_df is your DataFrame
train_df = pd.read_csv('train.csv')  # Adjust path as necessary
X_train = train_df.drop(['PassengerId', 'Survived'], axis=1)
y_train = train_df['Survived']

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Applying SelectKBest after preprocessing
# Adjust k as necessary. For example, k=10 selects the 10 best features
feature_selection = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('selectkbest', SelectKBest(chi2, k=10))])

feature_selection.fit(X_train, y_train)

# Getting support of selected features
support_mask = feature_selection.named_steps['selectkbest'].get_support()

# Transforming feature names
all_features = numeric_features.tolist() + \
               feature_selection.named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features).tolist()

selected_features = [all_features[i] for i in range(len(support_mask)) if support_mask[i]]

print("Selected Features:")
for feature in selected_features:
    print(feature)
