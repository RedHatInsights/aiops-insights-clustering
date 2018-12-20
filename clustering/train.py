import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import multiprocessing as mp

class Cluster:
    def __init__(self, pd_rules, index_cols=None,
                 categorical_cols=None, drop_cols=None,
                 n_clusters_low=2, n_clusters_high=50, n_clusters_stepsize=5, n_processes=None):

        self.data = pd_rules.copy()
        self.index_cols = index_cols

        #preprocessing
        self.categorical_cols = categorical_cols
        self.drop_cols = drop_cols
        self.preprocessed = False

        self.data_for_kmeans = None  # zscore -> pca -> zscore

        #clustering scan
        self.n_clusters_low = n_clusters_low
        self.n_clusters_high = n_clusters_high
        self.n_clusters_stepsize = n_clusters_stepsize
        self.n_processes = None

        self.models_dict = None
        self.inertia_dict = None

        #clustering optimal
        self.n_clusters_optimal = None
        self.kmeans_optimal = None


    def train_the_cluster(self):
        self.preprocess() #modify self.data

        preprocess_dict = self.preprocess_prepare_for_clustering(pca_variance_threshold=0.99)

        if self.n_processes is None:
            self.find_nclusters()
        else:
            self.find_nclusters_parallel()

        self.find_elbow()  # hard-coded currently
        self.train_cluster()

        return {
            "prepca_scaler": preprocess_dict["prepca_scaler"],
            "postpca_scaler": preprocess_dict["postpca_scaler"],
            "pca": preprocess_dict["pca"],
            "kmeans": self.kmeans_optimal
        }


    def preprocess(self):
        if not self.preprocessed: #so train_the_cluster can be called multiple times if necessary
            self.preprocess_index()
            self.preprocess_dropcols()
            self.preprocess_onehot()
            self.preprocess_fillnulls()
        self.preprocessed = True

    def preprocess_index(self):
        if self.index_cols is not None:
            self.data.set_index(self.index_cols, inplace=True)

    def preprocess_onehot(self):
        '''one-host encoding
        '''
        if self.categorical_cols is not None:
            df_rest = self.data.drop(self.categorical_cols, axis=1)  # non-categorical columns

            df_list = []
            for col in self.categorical_cols:
                df_onehot = pd.get_dummies(self.data[col], prefix=col)
                df_list.append(df_onehot)
            df_onehot = pd.concat(df_list, axis=1)  # concat all one-hot encoded cols

            df_onehot = pd.concat([df_onehot, df_rest], axis=1)  # concat non-categorical cols

            self.data = df_onehot

    def preprocess_dropcols(self):
        '''Only keep rules columns
        Makes assumptions about data columns
        '''
        if self.drop_cols is not None:
            self.data.drop(self.drop_cols, axis=1, inplace=True)

    def preprocess_fillnulls(self):
        '''Make more sophisticated
        '''
        self.data.fillna(0, inplace=True)

    def preprocess_pca(self, df, pca_variance_threshold=0.99):
        pca = PCA()

        df_pca = pd.DataFrame(pca.fit_transform(df))
        n_components = self.find_pca_ncomponents(pca, pca_variance_threshold)
        df_pca = df_pca.iloc[:, :n_components]
        pca.n_components = n_components

        return pca, df_pca

    def find_pca_ncomponents(self, pca_model, pca_variance_threshold):
        variance_cumsum = np.cumsum(pca_model.explained_variance_ratio_)

        gt_threshold = variance_cumsum[variance_cumsum > pca_variance_threshold]
        if len(gt_threshold) == 0:
            n_components = len(pca_model.explained_variance_ratio_)
        else:
            n_components = np.where(variance_cumsum == gt_threshold[0])[0][0]

        return n_components

    def preprocess_normalize(self, df):
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

        return scaler, df_scaled

    def preprocess_prepare_for_clustering(self, pca_variance_threshold=0.99):
        scaler_prepca, df_scaled = self.preprocess_normalize(self.data)

        pca, df_scaled_pca = self.preprocess_pca(df_scaled, pca_variance_threshold=pca_variance_threshold)

        scaler_postpca, df_scaled_pca_scaled = self.preprocess_normalize(df_scaled_pca)

        self.data_for_kmeans = df_scaled_pca_scaled

        return {'prepca_scaler': scaler_prepca,
                'pca': pca,
                'postpca_scaler': scaler_postpca,
                'df_prepca_scaled': df_scaled,
                'df_pca': df_scaled_pca,
                'df_postpca_scaled': df_scaled_pca_scaled
                }

    def find_nclusters(self):
        if self.data_for_kmeans is None:
            raise AttributeError("Please call preprocess_prepare_for_clustering to prepare data for clustering.")

        self.inertia_dict, self.models_dict = {}, {}
        for ncl in range(self.n_clusters_low, self.n_clusters_high, self.n_clusters_stepsize):
            kmeans = KMeans(n_clusters=ncl)
            kmeans.fit(self.data_for_kmeans)

            self.inertia_dict[ncl] = kmeans.inertia_
            self.models_dict[ncl] = kmeans

    def find_nclusters_parallel(self):
        manager = mp.Manager()
        models_shared_dict = manager.dict()
        inertia_shared_dict = manager.dict()

        proc_list = []
        counter = 0
        def run(data, ncl):
            kmeans = KMeans(n_clusters=ncl)
            kmeans.fit(data)

            inertia_shared_dict[ncl] = kmeans.inertia_
            models_shared_dict[ncl] = kmeans

        for ncl in range(self.n_clusters_low, self.n_clusters_high, self.n_clusters_stepsize):
            proc = mp.Process(target=run, args=(self.data, ncl,))
            proc.start()
            proc_list.append(proc)
            counter += 1

            if counter % self.n_processes == 0:
                [p.join() for p in  proc_list]
                proc_list = []

        [p.join() for p in proc_list]

        self.inertia_dict = dict(inertia_shared_dict)
        self.models_dict = dict(models_shared_dict)
    
    def find_elbow(self):
        '''check for off-by-one errors
        '''
        if self.inertia_dict is None:
            raise AttributeError("Please run find_nclusters to populated inertia_dict.")
        
        keys = np.sort(list(self.inertia_dict.keys()))
        values = np.array([self.inertia_dict[k] for k in keys])

        N = len(keys)

        thresholds = np.arange(2, N-1) #indices to slice at

        threshold_cuts, score_means = [], []

        for t in thresholds:
            model1, model2 = LinearRegression(), LinearRegression()

            domain1 = np.arange(t)
            domain2 = np.arange(t, N)

            model1.fit(domain1.reshape(-1,1), values[domain1])
            model2.fit(domain2.reshape(-1,1), values[domain2])

            score1 = model1.score(domain1.reshape(-1,1), values[domain1])
            score2 = model2.score(domain2.reshape(-1,1), values[domain2])

            score = (score1+score2)/2.0
            
            threshold_cuts.append(t)
            score_means.append(score)

        self.n_clusters_optimal = threshold_cuts[np.argmax(score_means)]

        return threshold_cuts, score_means

    def train_cluster(self):
        if self.data_for_kmeans is None:
            raise AttributeError("Please call preprocess_prepare_for_clustering to prepare data for clustering.")

        if not self.n_clusters_optimal:
            raise AttributeError("Please run find_n_clusters and find_elbow to find optimal n_clusters.")

        self.kmeans_optimal = KMeans(n_clusters=self.n_clusters_optimal).fit(self.data_for_kmeans)

    def interpret(self):
        if self.kmeans_optimal is None:
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