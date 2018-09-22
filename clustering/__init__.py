import pandas as pd
from sklearn.cluster import KMeans

def cluster(pd_rules: pd.DataFrame) -> dict:
    """
    Cluster data

    Normalize data and than run KMeans on it (categorize into 5 clusters)
    """
    data = ~pd_rules.isin([False, 'False', None, 'None', 0])
    clusters = KMeans(n_clusters=5).fit_predict(data)
    return dict(zip(pd_rules['system_id'], clusters))
