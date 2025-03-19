import numpy as np
from scipy.spatial import distance
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def rotateWindow(X,newSample):

    siz = X.shape[0]
    for i in range(siz-1):
        X[i] = X[i+1]
    X[siz-1] = newSample

    return X

def atualizaJanela(novaAmostra, centros, tempo_real_bins, Epslon, X,_index_clusteres):

    ind_closest = closest_node(novaAmostra, centros)

    if (distance.euclidean(novaAmostra[0], centros[ind_closest]) > Epslon):
        tempo_real_bins[-1] += 1
        ind_closest = -1

    else:
        tempo_real_bins[ind_closest] += 1

    tempo_real_bins[_index_clusteres[0]] -= 1
    rotateWindow(_index_clusteres, ind_closest)

    X = rotateWindow(X, novaAmostra)

    return X, tempo_real_bins, _index_clusteres