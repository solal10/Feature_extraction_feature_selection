import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from dataprep_tomer import prepare_data


def apply_pca(X, n_components=7):
    """
    Function to apply PCA to the data and plot the cumulative variance explained

    :param X: The data
    :param n_components: The number of principal components to keep
    :return: None
    """
    # Initialize the PCA object
    pca = PCA(n_components=n_components)

    # Fit and transform the data
    pca.fit(X)

    # The amount of variance that each PC explains
    var = pca.explained_variance_ratio_

    # Cumulative Variance explains
    var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)

    # Plot the cumulative variance explained
    plt.plot(var1, marker='o', linestyle='-', color='b')

    # Add dots and values
    for i, txt in enumerate(var1):
        plt.annotate(f'{txt:.2f}%', (i, txt), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs. Number of Principal Components')
    plt.show()


if __name__ == "__main__":
    # Prepare data
    X, y = prepare_data()
    X = scale(X)

    X1 = X.copy()

    # Apply PCA
    apply_pca(X1, X1.shape[1])

    # Choose the amount of Principal Components to keep
    PC_Amount = int(input("Choose the amount of Principal Components to keep: "))

    # Initialize the PCA object
    pca = PCA(n_components=PC_Amount)

    # The transformed data after applying PCA to the original data with the chosen amount of Principal Components
    X_new = pca.fit_transform(X)

    print(X_new)
