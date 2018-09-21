import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def cluster(pd_rules: pd.DataFrame) -> dict:
    """
    Cluster data

    Normalize data and than run KMeans on it (categorize into 5 clusters)
    """
    pd_allin = ~pd_rules.isin([False, 'False', None, 'None', 0])

    # TODO: Fix me
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
