import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def _create_incidents_matrix(dataframe):
    false_values = [False, 'False', None, 'None', 0]
    # Map values to binary
    transformed = np.isin(dataframe.values, false_values, invert=True)
    # Add counts of incident labels set to 1
    counts = np.count_nonzero(dataframe.loc[:, dataframe.columns.str.startswith('i_')], axis=1)
    counts = counts.reshape(counts.shape[0],-1)
    # Add the counts as a last column for each entry
    return np.hstack((transformed, counts))

def cluster(pd_rules):
    np_allin = _create_incidents_matrix(pd_rules)

    # TODO: Fix me
    pd_allin = pd.DataFrame(data=np_allin[:, :-1], columns=allin_labels)
    # Kmeans clustering model pipeline
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(pd_allin)
    data = pd.DataFrame(np_scaled)
    pca = PCA(n_components=3)
    data = pca.fit_transform(data)
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)
    # Choose the k by using elbow method
    n_cluster = range(1, 50)
    kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
    # The ultimate output: system with its associated cluster id -- pd_allin['cluster']
    pd_allin['cluster'] = kmeans[4].predict(data)
    # pd_allin['principal_feature1'] = data[0]
    # pd_allin['principal_feature2'] = data[1]
    pd_allin['system_id'] = pd_rules['system_id']
    return dict(zip(pd_allin['system_id'], pd_allin['cluster']))
