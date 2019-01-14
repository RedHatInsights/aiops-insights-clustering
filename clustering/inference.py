import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

from .train import Cluster

'''
Save n_components in pca object and get rid of get_pca_ncomponents
Add tests:
    ???
Ensure any df going to predict goes through preprocess
'''

class Inference(Cluster):
    def __init__(self, pd_rules, index_cols=None, \
                 categorical_cols=None, drop_cols=None,
                 models_dict=None, n_processes=None):
        
        super().__init__(pd_rules, index_cols=index_cols, categorical_cols=categorical_cols, drop_cols=drop_cols)
        self.n_processes = n_processes

        super().preprocess()

        self.models_dict = models_dict
        self.load_models()


    def load_models(self):
        if self.models_dict is None:
            raise AttributeError("Please specify pre-trained models to load")

        self.prepca_scaler = self.models_dict['prepca_scaler']
        self.pca = self.models_dict['pca']
        self.postpca_scaler = self.models_dict['postpca_scaler']
        self.kmeans_optimal = self.models_dict['kmeans']

    def preprocess_prepare_for_clustering(self):
        df_prepca_scaled = self.prepca_scaler.transform(self.data)       
        
        df_pca = pd.DataFrame(self.pca.transform(df_prepca_scaled)).iloc[:,:self.pca.n_components]

        df_postpca_scaled = self.postpca_scaler.transform(df_pca)

        self.data_for_kmeans = df_postpca_scaled

    def predict(self):      
        if self.data_for_kmeans is None:
            self.preprocess_prepare_for_clustering()

        cl = self.kmeans_optimal.predict(self.data_for_kmeans)

        # Conversions required to support JSON serialization:
        # Index just by one level of index
        # Convert using numpy.narray.tolist(), to ensure Python native types
        index = self.data.index.get_level_values(0).tolist()
        cl = cl.tolist()
        cl_dict = dict(zip(index, cl))

        return cl_dict

    def predict_proba(self, type='inverse'):
        if self.data_for_kmeans is None:
            self.preprocess_prepare_for_clustering()        

        if type!='inverse' and type!='exponential':
            raise ValueError("type should be 'inverse' or 'exponential'")

        #compute pairwise distances
        pair_dist = pairwise_distances(self.data_for_kmeans, self.kmeans_optimal.cluster_centers_, metric='euclidean', n_jobs=self.n_processes)
        pair_dist = pair_dist.astype(float)

        #compute probabilities
        if type=='inverse':
            if np.sum(pair_dist==0)>0:
                pair_dist[pair_dist==0] = 1e-10
            
            probs = np.apply_along_axis(lambda x: x/np.sum(x), 1, 1./pair_dist)
        elif type=='exponential':
            probs = np.apply_along_axis(lambda x: x/np.sum(x), 1, np.exp(-pair_dist))

        #check all probabilities add up to 1
        assert(int(probs.sum())==probs.shape[0])

        #check max prob cluster is same as prediction from self.predict
        argmax = np.apply_along_axis(np.argmax, 1, probs)
        cl = self.kmeans_optimal.predict(self.data_for_kmeans)
        assert(np.sum(argmax==cl)==probs.shape[0])

        #some munging to create output dict
        prob_list = np.apply_along_axis(lambda x: dict(zip(np.arange(self.kmeans_optimal.n_clusters), x)), 1, probs) 
        results = dict(zip(self.data.index, prob_list))

        return results