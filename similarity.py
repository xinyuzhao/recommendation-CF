import math


def euclidean_distance (X, Y):
    """
    Measure similarity between X and Y using euclidean distance. 
    
    Parameters
    ----------
    X: vector, shape (1, n_features)
    Y: vector, sahpe (1, n_features)

    Returns
    -------
    similarity: float
    """

    sim = set() 
    for f in X:
        if f in Y:
            sim.add(f)

    # if X and Y has no item in common
    if len(sim) == 0:
        return 0

    return 1 / (1 + sum((X[f] - Y[f]) ** 2 for f in sim))

def pearson_distance (X, Y):
    """
    Measure similarity between X and Y using euclidean distance. 
    
    Parameters
    ----------
    X: vector, shape (1, n_features)
    Y: vector, sahpe (1, n_features)

    Returns
    -------
    similarity: float
    """

    sim = set() 
    for f in X:
        if f in Y:
            sim.add(f)

    n = len(sim)

    if not n:
        return 0
    
    sum1 = sum(X[f] for f in sim)
    sum2 = sum(Y[f] for f in sim)
    
    sum1Sq = sum(X[f] * X[f] for f in sim)
    sum2Sq = sum(Y[f] * Y[f] for f in sim)

    pSum = sum (X[f] * Y[f] for f in sim)

    num = pSum - (sum1 * sum2 / n)
    den = math.sqrt((sum1Sq - sum1 * sum1 / n) * (sum2Sq - sum2 * sum2 / n))
    
    return 0 if den == 0 else num / den

