import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def knn_acc(embeddings, labels, test_embeddings = None, test_labels = None,  test_size=0.1, k = 10, rs=42, metric="euclidean"):
    """Computes kNN accuracy.
    
    Parameters
    ----------
    embeddings : array-like
        Full data if `test_embeddings`=None, or only train split if test split is also passed in `test_embeddings`.
    labels : array-like
        Full labels if `test_embeddings`=None, or only train split if test split is also passed in `test_labels`.
    test_embeddings  : array-like, default = None
        Test set (optional, if not passed a test set will be created).
    test_labels : array-like, default = None
        Test split of the labels (optional, if not passed a test split will be created).
    test_size :  float, default=0.1
        Fraction of the data used for the test set.
    k : int, default=10
        Number of nearest neighbors to use.
    rs : int, default=42
        Random seed.
    metric :  string, {"euclidean", "cosine"}
        Metric used to compute distances to build the kNN graph. You can pass whatever KNeighborsClassifier from sklearn accepts.
    
    Returns
    -------
    knn_accuracy_score : float
        kNN accuracy.
    
    """
    # check whether both test_embeddings and test_labels were not passed
    assert (
        not test_embeddings is None or test_labels is None
    ), "You did not pass a test set (so the data will be split in train/test sets), but you passed a test split for the labels."
    

    # create/assign test splits
    if test_embeddings is None:
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, random_state = rs)
    else:
        X_train, X_test, y_train, y_test = embeddings, labels, test_embeddings, test_labels

    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=-1, metric=metric)
    knn = knn.fit(X_train, y_train)
    knn_accuracy_score =  knn.score(X_test, y_test)
    return knn_accuracy_score


def knn_accuracy(embeddings, labels, test_embeddings = None, test_labels = None, test_size=0.1, k = 10, rs=42, metric="euclidean"):
    """
    Expand knn_acc function to also accept several representations of a dataset (i.e., with the same labels) to evaluate, passed as a list of datasets [X1,X2,...,XN].
    It returns the accuracies in one list: for a single dataset as one single value, for several datasets as several values in the same list.
    """

    if not isinstance(embeddings, list):
        embeddings = [embeddings]
        test_embeddings = [test_embeddings]
    else:
        if test_embeddings is not None:
            assert len(embeddings) == len(test_embeddings), "Train and test splits don't have the same number of representations to evaluate."
        
    knn_accuracy_values = []
    for i, embed in enumerate(embeddings):
        knn_accuracy_values.append(knn_acc(embed, labels, test_embeddings = test_embeddings[i], test_labels = test_labels, test_size=test_size, k = k, rs=rs, metric=metric))
        
    knn_accuracy_values= np.array(knn_accuracy_values)
    
    return knn_accuracy_values




def knn_accuracy_whitening_scores(X, y, rs=42):
    """Calculates kNN accuracy of raw, centered and whitened data.
    It calculates it for to distance metrics: cosine and euclidean.

    Parameters
    ----------
    X : list of array-like
        List with the different datasets for which to calculate the kNN accuracy.
    y : array-like
        Array with labels (colors).
    rs : int, default=42
        Random seed.

    Returns
    -------
    scores : array of floats of shape (3,2)
        Array with the kNN accuracy for the different distance metrics and versions of the data.

    """

    n = X.shape[0]
    Xcentered = X - np.mean(X, axis=0)
    Xwhitened = PCA(whiten=True).fit_transform(X)

    scores = np.zeros((3, 2))

    for j, metric in enumerate(["euclidean", "cosine"]):

        acc = knn_accuracy([X, Xcentered, Xwhitened], y, metric=metric, rs=rs)
        scores[:, j] = acc

    return scores




def lin_acc(embeddings, labels, test_embeddings = None, test_labels = None,  test_size=0.1, rs=42):
    """Computes linear accuracy.
    
    Parameters
    ----------
    embeddings : array-like
        Full data if `test_embeddings`=None, or only train split if test split is also passed in `test_embeddings`.
    labels : array-like
        Full labels if `test_embeddings`=None, or only train split if test split is also passed in `test_labels`.
    test_embeddings  : array-like, default = None
        Test set (optional, if not passed a test set will be created).
    test_labels : array-like, default = None
        Test split of the labels (optional, if not passed a test split will be created).
    test_size :  float, default=0.1
        Fraction of the data used for the test set.
    rs : int, default=42
        Random seed.
    
    Returns
    -------
    linear_accuracy : float
        Linear accuracy.
    
    """
    # check whether both test_embeddings and test_labels were not passed
    assert (
        not test_embeddings is None or test_labels is None
    ), "You did not pass a test set (so the data will be split in train/test sets), but you passed a test split for the labels."


    # create/assign test splits
    if test_embeddings is None:
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, random_state = rs)
    else:
        X_train, X_test, y_train, y_test = embeddings, labels, test_embeddings, test_labels

    log_reg = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty=None,
                solver="saga",
                tol=1e-2,
                random_state=rs,
                n_jobs=-1,
                max_iter=1000,
            ),
        )
    
    log_reg.fit(X_train, y_train)
    linear_accuracy_score = log_reg.score(X_test, y_test)
    return linear_accuracy_score



def linear_accuracy(embeddings, labels, test_embeddings = None, test_labels = None, test_size=0.1, rs=42):
    """
    Expand lin_acc function to also accept several representations of a dataset (i.e., with the same labels) to evaluate, passed as a list of datasets [X1,X2,...,XN].
    It returns the accuracies in one list: for a single dataset as one single value, for several datasets as several values in the same list.
    """
    
    if not isinstance(embeddings, list):
        embeddings = [embeddings]
        test_embeddings = [test_embeddings]
    else:
        if test_embeddings is not None:
            assert len(embeddings) == len(test_embeddings), "Train and test splits don't have the same number of representations to evaluate."
        
    lin_accuracy = []
    for i, embed in enumerate(embeddings):
        lin_accuracy.append(lin_acc(embed, labels, test_embeddings = test_embeddings[i], test_labels = test_labels, test_size=test_size, rs=rs))
        
    lin_accuracy= np.array(lin_accuracy)
    
    return lin_accuracy