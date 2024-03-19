from collections import Counter
from dataprep_tomer import prepare_data
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def train_and_evaluate(X_train, X_test, y_train, y_test, max_iter=1000):
    """
    Train a Logistic Regression model with increased max_iter and evaluate its performance.
    :param X_train: Training features
    :param X_test: Testing features
    :param y_train: Training labels
    :param y_test: Testing labels
    :param max_iter: Maximum number of iterations for Logistic Regression
    :return: Accuracy of the trained model on the test set
    """
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


def find_best_k(X, y, scoring_func, num_iterations=10):
    """
    Find the best k for the given scoring function by training a simple classifier and evaluating its performance.
    :param X: The feature matrix
    :param y: The target variable
    :param scoring_func: A scoring function from sklearn.feature_selection module
    :param num_iterations: Number of iterations to find the best k
    :return: The most frequent k found in the iterations with the given scoring function
    """
    max_k = X.shape[1]
    best_k_counter = Counter()
    print(f"Finding the best k for {scoring_func.__name__}...")

    for _ in range(num_iterations):
        print(f"Iteration {_ + 1}/{num_iterations}")
        current_best_k = None
        best_accuracy = 0.0

        for k in range(1, max_k + 1):
            selector = SelectKBest(score_func=scoring_func, k=k)
            X_new = selector.fit_transform(X, y)

            # Scale the data using StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_new)

            # Split the scaled data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Train a simple classifier and evaluate
            accuracy = train_and_evaluate(X_train, X_test, y_train, y_test)

            # Update current_best_k and best_accuracy if the current k is better
            if accuracy > best_accuracy:
                current_best_k = k
                best_accuracy = accuracy

        # Increment the counter for the current_best_k
        best_k_counter[current_best_k] += 1

    # Get the most frequent k from the counter
    most_frequent_k, _ = best_k_counter.most_common(1)[0]

    return most_frequent_k


def select_features(X, y, scoring_functions=[f_classif, mutual_info_classif, chi2]):

    """
    Select the best features using the given scoring functions and find the best k for each scoring function.
    :param X: The feature matrix
    :param y: The target variable
    :param scoring_functions: The scoring functions to use for feature selection
    :return: A dictionary containing the selected features and best k for each scoring function
    """

    selected_features_dict = {}

    for scoring_func in scoring_functions:
        print(f"Selecting features using {scoring_func.__name__}...")

        # Find the best k for the current scoring function
        best_k = find_best_k(X, y, scoring_func)

        # Initialize the SelectKBest object with the best k
        selector = SelectKBest(score_func=scoring_func, k=best_k)

        # Fit the selector to your data
        X_new = selector.fit_transform(X, y)

        # Get the selected feature indices
        selected_indices = selector.get_support(indices=True)

        # Store the selected features
        selected_features = X.columns[selected_indices]
        selected_features_dict[scoring_func.__name__] = {"best_k": best_k, "features": selected_features, "X_new": X_new}

    return selected_features_dict


def represent_results(selected_features_dict):
    """
    Print the selected features and best k for each scoring function
    :param selected_features_dict: The dictionary containing the selected features and best k for each scoring function
    :return: None
    """
    for scoring_func, results in selected_features_dict.items():
        print(f"Best k for {scoring_func}: {results['best_k']}")
        print(f"Selected Features using {scoring_func}:", results['features'])
        print("Transformed data:")
        print(results['X_new'])


if __name__ == "__main__":
    # Prepare data
    X, y = prepare_data()

    # Select features
    selected_features_dict = select_features(X, y)

    # Represent the results
    represent_results(selected_features_dict)
