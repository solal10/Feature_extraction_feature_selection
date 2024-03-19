import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from dataprep_tomer import prepare_data


def random_forest_feature_importance(X, y):
    """
    Perform Random Forest feature importance with hyperparameter tuning:
    1. Define the parameter grid for grid search
    2. Create a Random Forest Classifier
    3. Initialize GridSearchCV
    4. Fit the model to your data
    5. Get the best parameters
    6. Use the best parameters to train the model
    7. Get feature importances
    8. Get the indices of features sorted by importance
    9. Plot the importance of each feature

    :param X: The feature matrix
    :param y: The target column
    :return: The indices of features sorted by importance and the best hyperparameters found
    """
    # Define the parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Create a Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the model to your data
    grid_search.fit(X, y)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Use the best parameters to train the model
    best_rf_classifier = RandomForestClassifier(**best_params, random_state=42)
    best_rf_classifier.fit(X, y)

    # Get feature importances
    feature_importances = best_rf_classifier.feature_importances_

    # Get the indices of features sorted by importance
    sorted_indices = feature_importances.argsort()[::-1]

    # Plot the importance of each feature
    plot_feature_importance(X.columns, feature_importances, sorted_indices)

    return sorted_indices, best_params


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
    plt.ylabel('Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.show()


if __name__ == "__main__":
    # Prepare data
    X, y = prepare_data()
    print(X,y)
    # Perform Random Forest feature importance with hyperparameter tuning
    top_indices, best_params = random_forest_feature_importance(X, y)
    print("Best Hyperparameters:", best_params)

    # Ask the user to choose the amount of features to keep
    choice_of_k = int(input("Choose the amount of features to keep: "))

    # Get the selected features
    new_columns = X.columns[top_indices[:choice_of_k]]

    # Get the new data with the selected features
    new_data = X[new_columns]
    print(new_data.head())
