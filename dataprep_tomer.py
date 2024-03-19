import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

def prepare_data(file_path='train.csv'):
    # Load the dataset from CSV file
    data = pd.read_csv(file_path)

    # Use LabelEncoder for categorical columns
    label_encoder = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Save the target variable before imputing
    # Adjust the index as necessary; here it's assumed the target column's index is 11
    y = data.iloc[:, 11].values  # Using .values to ensure y is a NumPy array

    # Drop the target column from the dataset before imputing
    data = data.drop(columns=data.columns[11])

    # Apply imputation to the remaining data
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_imputed = imputer.fit_transform(data)

    # Convert the imputed data back to a DataFrame (optional step if you need dataframe functionalities)
    data_imputed_df = pd.DataFrame(data_imputed, columns=data.columns)

    # Now data_imputed (or data_imputed_df if converted to DataFrame) can be used as your features matrix
    X = data_imputed_df  # Use this line if you prefer to work with DataFrames
    # Or simply use data_imputed if you're okay with working with NumPy arrays

    return X, y
