import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pyarrow.parquet as pq
import s3fs
import os


def pandas_dataframe():
    (ceph_key, ceph_secret, ceph_host) = (os.getenv('DH_CEPH_KEY'), os.getenv('DH_CEPH_SECRET'), os.getenv('CEPH_HOST'))
    client_kwargs = { 'endpoint_url' : ceph_host }

    if not ceph_key:
        # This is to run local
        return pq.ParquetDataset('./data/').read_pandas().to_pandas()
    else:
        s3 = s3fs.S3FileSystem(secret=ceph_secret, key=ceph_key, client_kwargs=client_kwargs)
        return pq.ParquetDataset('DH-DEV-INSIGHTS/2018-06-01/rule_data', filesystem=s3).read_pandas().to_pandas()

def find_incident(labels):
    res = set()
    for i in range(len(labels)):
        if labels[i].startswith('i_'):
            res.add(i)
    return res

# numpy array construction
def construct_np_allin(allin):
    np_allin = []
    for i in range(m):
        tmp = []
        count = 0
        for j in range(n):
            if allin[i][j] == True:
                tmp.append(1)
            elif allin[i][j] == False:
                tmp.append(0)
            elif allin[i][j] != 'None':
                tmp.append(1)
            else:
                tmp.append(0)
            if j in incident_labels and allin[i][j] == 1:
                count += 1
        np_allin.append(tmp+[count])
    return np_allin

pd_rules = pandas_dataframe()
np_rules = pd_rules.as_matrix()
sysid = np_rules[:, 1]
labels = list(pd_rules.columns.values)
allin = np_rules[:, 3:]
allin_labels = labels[3:]
incident_labels = find_incident(allin_labels)
m, n = allin.shape
np_allin = construct_np_allin(allin)
np_allin = np.array(np_allin)
pd_allin = pd.DataFrame(data = np_allin[:, :-1], columns = allin_labels)


# Kmeans clustering model pipeline

min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(pd_allin)
data = pd.DataFrame(np_scaled)
pca = PCA()
pca.fit(data)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
pca = PCA(n_components=3)
data = pca.fit_transform(data)
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)


# Choose the k by using elbow method

n_cluster = range(1, 50)
kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]

# The ultimate output: system with its associated cluster id -- pd_allin['cluster']

pd_allin['cluster'] = kmeans[4].predict(data)
pd_allin['principal_feature1'] = data[0]
pd_allin['principal_feature2'] = data[1]
pd_allin['cluster'].value_counts()


print(pd_allin['cluster'])
