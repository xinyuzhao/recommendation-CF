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

def topMatches (data, sample, N, simFunc):
    """
    Return top N samples that have highest similarity with the given sample 
    
    Parameters
    ----------
    data: dictionary of M dictionaries 
    sample: dictionary 
    N: int, <= M
    simFunc: similarity measures (euclidean distantance or pearson distance)

    Returns
    -------
    scores[0:N]: top N samples with corresponding similarity scores 
    """

    scores = [] 
    for key in data.keys():
        if key != sample:
            scores.append((simFunc(data[key], data[sample]), key))

    scores.sort(reverse=True)

    return scores[0 : N]

def getKey(item):
    return item[1]

def getRecommendation (data, sample, N=5, simFunc=pearson_distance):
    """
    Compute the similarity between the sample and others in the dataset. 
    Then, recommend items that have not been known by the sample.
    
    Parameters
    ----------
    data: dictionary of M dictionaries 
    sample: dictionary 
    N: int, <= M
    simFunc: similarity measures (euclidean distantance or pearson distance)

    Returns
    -------
    output: a list of recommended items with corresponding scores 
    """

    if N > len(data):
        raise ValueError('N must be smaller than or equal to sample size')

    matches = topMatches(data, sample, N, simFunc)

    totals = {}
    sim_sum = {}

    for other in matches:
        sim_score = other[0]
        name = other[1]

        for movie, rate in data[name].items():
            if movie not in data[sample]:
                if movie not in totals:
                    totals[movie] = rate * sim_score
                    sim_sum[movie] = sim_score
                else:
                    totals[movie] += rate * sim_score
                    sim_sum[movie] += sim_score

    output = []
    for key, value in totals.items():
        norm_score = totals[key] / sim_sum[key]
        output.append((key, norm_score))

    output.sort(reverse=True, key=getKey)

    return output

def transformDataSet(data):
    """
    Transform dataset between user-based and item-based
    
    Parameters
    ----------
    data: original dataset, a dictionary of dictionaries 

    Returns
    -------
    output: transformed dataset 
    """

    transformed = {}

    for key1 in data.keys():
        for key2, value in data[key1].items():
            if key2 not in transformed:
                transformed[key2] = {key1: value}
            else:
                transformed[key2][key1] = value

    return transformed





