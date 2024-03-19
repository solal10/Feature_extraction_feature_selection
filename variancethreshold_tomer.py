import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from dataprep_tomer import prepare_data


def variance_threshold_feature_importance(X):
    """
    Perform Variance Threshold feature importance:
    1. Initialize the Variance Threshold selector
    2. Fit the selector to your data
    3. Get the variances of each feature
    4. Sort the features by variance in descending order
    5. Plot the importance of each feature

    :param X: The feature matrix
    :return: The indices of features sorted by variance
    """
    # Initialize the Variance Threshold selector
    variance_selector = VarianceThreshold()

    # Fit the selector to your data
    variance_selector.fit(X)

    # Get the variances of each feature
    feature_variances = variance_selector.variances_

    # Sort the features by variance in descending order
    sorted_indices = feature_variances.argsort()[::-1]

    # Print or use the selected features
    selected_features = X.columns[sorted_indices]

    # Plot the importance of each feature
    plot_feature_importance(selected_features, feature_variances, sorted_indices)

    return sorted_indices


def plot_feature_importance(columns, importances, indices):
    """
    Plot the importance of each feature

    :param columns: The feature names
    :param importances: The feature importances
    :param indices: The indices of features sorted by importance
    :return: None
    """
    plt.figure(figsize=(10, 7))
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), columns[indices], rotation='vertical')
    plt.xlabel('Features')
    plt.ylabel('Variance')
    plt.title('Feature Importance (Variance)')
    plt.show()


if __name__ == "__main__":
    # Prepare data
    X, _ = prepare_data()

    # Perform Variance Threshold feature importance
    top_indices = variance_threshold_feature_importance(X)  # , threshold)

    # Choose the amount of features to keep
    choice_of_k = int(input("Choose the amount of features to keep: "))

    # Get the selected features
    new_columns = X.columns[top_indices[:choice_of_k]]
    print(new_columns)

    # Get the new data with the selected features
    new_data = X[new_columns]
    print(new_data.head())
