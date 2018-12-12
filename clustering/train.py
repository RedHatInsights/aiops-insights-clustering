import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Cluster:
    def __init__(self, pd_rules,
                 n_clusters_low=2, n_clusters_high=50, n_clusters_stepsize=5,
                 categorical_cols=None,
                 drop_cols=None):
        self.data = pd_rules

        self.data_for_kmeans = None  # zscore -> pca -> zscore

        self.n_clusters_low = n_clusters_low
        self.n_clusters_high = n_clusters_high
        self.n_clusters_stepsize = n_clusters_stepsize

        self.models_dict = None
        self.inertia_dict = None

        self.n_clusters_optimal = None
        self.kmeans_optimal = None

        self.categorical_cols = categorical_cols
        self.drop_cols = drop_cols

    def train_the_cluster(self):
        self.preprocess()

        # Train - prepare for clustering (z-score -> PCA -> z-score)
        df_cluster_dict = self.preprocess_prepare_for_clustering()
        # df_cluster_dict contains 3 models (two scalers and 1 PCA) and 3 dataframes (after each step)

        # scan n_clusters and find optimal point (In progress)
        data_for_kmeans = df_cluster_dict['df_postpca_scaled']
        models_dict, inertia_dict = self.find_nclusters(
            data_for_kmeans,
            self.n_clusters_low,
            self.n_clusters_high,
            self.n_clusters_stepsize
        )
        n_clusters_optimal = self.find_elbow(inertia_dict)  # hard-coded currently

        kmeans_optimal = self.train_cluster(data_for_kmeans, n_clusters_optimal)

        return {
            "prepca_scaler": df_cluster_dict["prepca_scaler"],
            "postpca_scaler": df_cluster_dict["postpca_scaler"],
            "pca": df_cluster_dict["pca"],
            "kmeans": kmeans_optimal
        }

    def preprocess(self):
        self.preprocess_dropcols()
        self.preprocess_onehot()
        self.preprocess_fillnulls()

    def preprocess_onehot(self):
        '''one-host encoding
        '''

        if self.categorical_cols is None:  # if list of categorical cols not passed, one-hot encode everything
            categorical_cols = self.data.columns
            df_rest = pd.DataFrame()
        else:
            df_rest = self.data.drop(self.categorical_cols, axis=1)  # non-categorical columns

        df_list = []
        for col in categorical_cols:
            df_onehot = pd.get_dummies(self.data[col], prefix=col)
            df_list.append(df_onehot)
        df_onehot = pd.concat(df_list, axis=1)  # concat all one-hot encoded cols

        df_onehot = pd.concat([df_onehot, df_rest], axis=1)  # concat non-categorical cols

        self.data = df_onehot

    def preprocess_dropcols(self):
        '''Only keep rules columns
        Makes assumptions about data columns
        '''
        if self.drop_cols:
            self.data.drop(self.drop_cols, axis=1, inplace=True)

    def preprocess_fillnulls(self):
        '''Make more sophisticated
        '''
        self.data.fillna(0, inplace=True)

    def preprocess_pca(self, df, pca_variance_threshold=0.99):
        pca = PCA()

        df_pca = pca.fit_transform(df)
        df_pca = pd.DataFrame(df_pca)

        variance_cumsum = np.cumsum(pca.explained_variance_ratio_)

        gt_threshold = variance_cumsum[variance_cumsum > pca_variance_threshold]
        if len(gt_threshold) == 0:
            n_components = len(pca.explained_variance_ratio_)
        else:
            n_components = np.where(variance_cumsum == gt_threshold[0])[0][0]

        return pca, df_pca, n_components

    def preprocess_normalize(self, df):
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

        return scaler, df_scaled

    def preprocess_prepare_for_clustering(self, pca_variance_threshold=0.99):
        scaler, df_scaled = self.preprocess_normalize(self.data)

        pca, df_pca, n_components = self.preprocess_pca(df_scaled, pca_variance_threshold=pca_variance_threshold)
        pca.n_components = n_components

        df_pca = df_pca.iloc[:, :n_components]

        scaler_postpca, df_scaled_pca_scaled = self.preprocess_normalize(df_pca)

        return {'prepca_scaler': scaler,
                'pca': pca,
                'postpca_scaler': scaler_postpca,
                'df_prepca_scaled': df_scaled,
                'df_pca': df_pca,
                'df_postpca_scaled': df_scaled_pca_scaled
                }

    def find_nclusters(self, df, n_clusters_low, n_clusters_high, n_clusters_stepsize):
        inertia_dict = {}
        models_dict = {}

        for ncl in range(n_clusters_low, n_clusters_high, n_clusters_stepsize):
            kmeans = KMeans(n_clusters=ncl)
            kmeans.fit(df)

            inertia_dict[ncl] = kmeans.inertia_
            models_dict[ncl] = kmeans

        return models_dict, inertia_dict

    def find_elbow(self, inertia_dict):
        return 5

    def train_cluster(self, df, n_clusters):
        model = KMeans(n_clusters=n_clusters)

        model.fit(df)

        return model

    def interpret(self):
        if not self.kmeans_optimal:
            raise AttributeError("Attribute kmeans_optimal not found. Please train model first.")

        return None
        '''
        cl = self.kmeans_optimal.predict(self.data_for_kmeans)

        df = pd.DataFrame(self.kmeans_optimal.cluster_centers_, columns = self.data_preprocess.columns)
        df['cl'] = np.arange(df.shape[0])

        model_dtree = DecisionTreeClassifier(max_depth=20)

        model_dtree.fit(df.drop('cl', axis=1), df['cl'])

        #TODO: generate descriptions from paths
        path = model_dtree.decision_path(df.drop('cl', axis=1))
        path_index = path.indices[path.indptr[0]:path.indptr[1]]

        def get_path(example_row, data, col_names, thresholds):
            traversed_nodes = path_nodes.indices[path_nodes.indptr[example_row]:path_nodes.indptr[example_row+1]]
            for node in traversed_nodes:
                print(f"Node hit: {features[node]} {col_names[node]} {100*thresholds[node]} {data[example_row][node]}")
        '''

    def train(self):
        # 1-hot and drop columns
        self.data_preprocess = self.preprocess(self.data)

        # z-score -> pca -> z-score
        df_cluster_dict = self.preprocess_prepare_for_clustering(self, self.data_preprocess,
                                                                 pca_variance_threshold=0.99)

        # df for clustering
        self.data_for_kmeans = df_cluster_dict['df_postpca_scaled']

        self.models_dict, self.inertia_dict = self.find_nclusters(self, self.data_for_kmeans, self.n_clusters_low,
                                                                  self.n_clusters_high, self.n_clusters_stepsize)

        self.n_clusters_optimal = self.find_elbow(self.inertia_dict)

        self.kmeans_optimal = self.train_cluster(self.data_for_kmeans, self.n_clusters_optimal)

        # replace by ceph store
        self.save_model(df_cluster_dict['prepca_scaler'], 'prepca_scaler.pkl')
        self.save_model(df_cluster_dict['postpca_scaler'], 'postpca_scaler.pkl')
        self.save_model(df_cluster_dict['pca'], 'pca.pkl')

        self.save_model(self.kmeans_optimal, "kmeans_optimal.pkl")

    def future_pipeline(self):
        '''Might move to pipelines in the future
        All intermediate models need to have transform implemented
        Final model needs to have predict implemented
        '''
        prepca_scaler = StandardScaler()
        pca = PCA()
        postpca_scaler = StandardScaler()
        cluster = KMeans()

        p = Pipeline([('prepca_scaler', prepca_scaler),
                      ('pca', pca),
                      ('postpca_scaler', postpca_scaler),
                      ('cluster', kmeans)])


def diff(x):
    xdiff = []
    for i in range(1, len(x)):
        xdiff.append(x[i] - x[i - 1])

    return xdiff


def run():
    '''
    Just for random testing
    '''
    cl = train.Cluster(df)
    cl.data_preprocess = cl.preprocess(cl.data, categorical_cols=categorical_cols, drop_cols=drop_cols)
    categorical_cols = ['Sex', 'Embarked']
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']

    cl.data_preprocess = cl.preprocess(cl.data, categorical_cols=categorical_cols, drop_cols=drop_cols)
    cl.data_preprocess = cl.preprocess_dropnulls(cl.data_preprocess)
    df_cluster_dict = cl.preprocess_prepare_for_clustering(cl.data_preprocess)
    df_postpca_scaled = df_cluster_dict['df_postpca_scaled']
    cl.data_for_kmeans = df_postpca_scaled
    cl.models_dict, cl.inertia_dict = cl.find_nclusters(cl.data_for_kmeans, 2, 20, 1)
    cl.n_clusters_optimal = 5
    cl.kmeans_optimal = cl.train_cluster(cl.data_for_kmeans, cl.n_clusters_optimal)

    cl.save_model(df_cluster_dict['prepca_scaler'], 'prepca_scaler.pkl')
    cl.save_model(df_cluster_dict['postpca_scaler'], 'postpca_scaler.pkl')
    cl.save_model(df_cluster_dict['pca'], 'pca.pkl')
    cl.save_model(cl.kmeans_optimal, 'kmeans.pkl')

    inf.predict(df, 'prepca_scaler.pkl', 'pca.pkl', 'postpca_scaler.pkl', 'kmeans.pkl')
