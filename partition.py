from k_means_constrained import KMeansConstrained
from sklearn.cluster import DBSCAN
import numpy as np

def relabel_clusters(cluster_array):
    unique_clusters = np.unique(cluster_array)
    unique_clusters = unique_clusters[unique_clusters != -1]
    cluster_map = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
    new_cluster_array = np.array([cluster_map[label] if label != -1 else -1 for label in cluster_array])
    return new_cluster_array

def realiza_Particao_Inicial(X, Epslon, amostras_minimas, n_bins):

    db = DBSCAN(eps=Epslon, min_samples=amostras_minimas).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    min_cluster_size = np.shape(X)[0]
    min_label = 0
    temp_labels = labels.copy()
    max_l = np.max(labels)
    for i in range(max_l + 1):
        if((labels == i).sum() < 2*n_bins):
            temp_labels[labels==i]=-1
    labels = relabel_clusters(temp_labels)
    max_l = np.max(labels)
    if (max_l == - 1):
        labels[:] = 0
    n_noise_ = list(labels).count(-1)
    _noise_index = (labels == -1)

    for i in range(np.max(labels) + 1):

        if (np.shape(X[labels == i])[0] < min_cluster_size and np.round(np.shape(X[labels == i])[0] / n_bins) > 0 and np.shape(X[labels == i])[0] >= 2*n_bins):
            min_cluster_size = np.shape(X[labels == i])[0]
            min_label = i

    bin_size = np.round(min_cluster_size / n_bins)

    filtered_X = X[labels != -1]

    _indexes_clusteres = labels[labels != -1]
    _indexes_clusteres_full = labels
    filtered_labels = labels[labels != -1]

    centros = np.empty(shape=[0, np.shape(X)[1]])
    bins = []
    _n_max_labels = 0
    for i in range(np.max(labels) + 1):

        X_grupo = filtered_X[filtered_labels == i]
        clf = KMeansConstrained(
            n_clusters=int(np.floor(np.shape(X_grupo)[0] / bin_size)),
            size_min=bin_size,
            #size_max=(bin_size + np.ceil(bin_size / 5)),
            random_state=0,
            max_iter = 10,
            n_jobs = -1
        )

        clf.fit_predict(X_grupo)

        _indexes_clusteres[filtered_labels==i] = clf.labels_ + (_n_max_labels)

        _n_max_labels = np.max(_indexes_clusteres+1)  # obtem o número máximo de labels por rodada do kmeans

        hist_bin = []
        for k in range(int(np.floor(np.shape(X_grupo)[0] / bin_size))):
            hist_bin.append(np.sum(clf.labels_ == k))

        bins.append(hist_bin)
        centros = np.concatenate([centros, clf.cluster_centers_])

    _indexes_clusteres_full[labels!=-1] = _indexes_clusteres
    bins.append(n_noise_)
    del db
    if(np.max(labels)>0):
        del clf
    return filtered_X, centros, bin_size, n_noise_, bins,_indexes_clusteres_full