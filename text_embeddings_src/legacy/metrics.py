import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def knn_accuracy(embeddings, true_labels, test_size=0.1, k = 10, rs=42, set_numpy = True, metric="euclidean"):
    """Calculates kNN accuracy.
    In principle should do the same as the function above, but the way of selecting the train and test set is differently.
    Actually, if you use the same random seed as in the function above it gives you the same result (at least for the default parameters of train_test_split sklearn version 1.0.2).
    
    
    Parameters
    ----------
    embeddings : list 
        List with the different datasets for which to calculate the kNN accuracy.
    true_labels : array-like
        Array with labels (colors).
    k : int, default=10
        Number of nearest neighbors to use.
    rs : int, default=42
        Random seed.
    
    Returns
    -------
    knn_accuracy : float
        kNN accuracy of the dataset.
    
    """
    
    random_state = np.random.seed(rs)

    if type(embeddings) == list:
        knn_accuracy = []
        for embed in embeddings:
            X_train, X_test, y_train, y_test = train_test_split(embed, true_labels, test_size=test_size, random_state = random_state)
    
            knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=-1, metric=metric)
            knn = knn.fit(X_train, y_train)
            knn_accuracy.append(knn.score(X_test, y_test))
        if set_numpy == True:
            knn_accuracy= np.array(knn_accuracy)
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(embeddings, true_labels, test_size=test_size, random_state = random_state)
        knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=-1, metric=metric)
        knn = knn.fit(X_train, y_train)
        knn_accuracy = knn.score(X_test, y_test)

    
    return knn_accuracy



def knn_recall(X, Z, k=10, test_size=0.1, rs=42):
    """Calculates kNN recall. 
    Calculates the kNN recall for `k`nearest neighbors.
    
    Parameters
    ----------
    X : {array-like, sparse matrix}
        High-dimensional data.
    Z : array-like
        List of different low-dimensional data. When computing for only one low-dimensional dataset, still needs to be a list of array-like: [Z].
    k : int, default=10
        Number of nearest eeighbors.
    subset_size : int, optional
        Size of the subset of the data, if desired.
    
    Returns
    -------
    knn_recall : array like 
        KNN recall
    
    See Also
    --------
    knn_recall_affinity_matrix, knn_recall_and_ratios
    
    
    Note
    ----
    KNN recall is by definition the fraction of preserved nearest neighbors from the high-dimensional version of the dataset to the low-dimensional one.

    """
    
    if test_size is not None:
        _, X_test, _, Z_test = train_test_split(X, Z, test_size=test_size, random_state = rs)
        
        # In this case we will have to query k+1 points, because
        # sklearn returns the query point itself as one of the neighbors
        k_to_query = k + 1
    else:
        # In this case we can query k points
        k_to_query = k 
    
    nbrs1 = NearestNeighbors(n_neighbors=k_to_query, algorithm='brute', n_jobs=-1).fit(X)
    ind1 = nbrs1.kneighbors(X=None if test_size is None else X_test,
                            return_distance=False)
        
    knn_recall = []
    #for num, Z in enumerate(Zs):
    print('.', end='')
    nbrs2 = NearestNeighbors(n_neighbors=k_to_query, algorithm='brute', n_jobs=-1).fit(Z)
    ind2 = nbrs2.kneighbors(X=None if test_size is None else Z_test,
                            return_distance=False)

    intersections = 0.0
    for i in range(ind1.shape[0]):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    
    if test_size is None:
        knn_recall.append(intersections / ind1.shape[0] / k)
    else:
        # it substracts the ind1.shape[0] because when you take a subset of the data
        # in the NearestNeighbors.kneighbors function, it takes the query point itself as
        # one of the neighbors, so you need to substract the intersection of a point with himself
        knn_recall.append((intersections - ind1.shape[0]) / ind1.shape[0] / k)
    
    
    return knn_recall


def logistic_accuracy(
    embeddings, true_labels, test_size=0.1, rs=42, set_numpy=True
):
    """Calculates logistic accuracy.

    Parameters
    ----------
    embeddings : list
        List with the different datasets for which to calculate the kNN accuracy.
    true_labels : array-like
        Array with labels (colors).
    test_size : float
        Fraction of the data to take as test set.
    rs : int, default=42
        Random seed.

    Returns
    -------
    accuracy : float
        Accuracy of the logistic classifier in the test set.

    """

    random_state = np.random.seed(rs)

    if type(embeddings) == list:
        accuracy = []
        for embed in embeddings:
            X_train, X_test, y_train, y_test = train_test_split(
                embed,
                true_labels,
                test_size=test_size,
                random_state=random_state,
            )
            lr = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty=None,
                    solver="saga",
                    tol=1e-2,
                    random_state=random_state,
                    n_jobs=-1,
                    max_iter=1000,
                ),
            )

            lr.fit(X_train, y_train)
            accuracy.append(lr.score(X_test, y_test))
        if set_numpy == True:
            accuracy = np.array(accuracy)

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            true_labels,
            test_size=test_size,
            random_state=random_state,
        )

        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty=None,
                solver="saga",
                tol=1e-2,
                random_state=random_state,
                n_jobs=-1,
                max_iter=1000,
            ),
        )

        lr.fit(X_train, y_train)
        accuracy = lr.score(X_test, y_test)

    return accuracy
