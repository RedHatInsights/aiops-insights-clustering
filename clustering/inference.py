import pandas as pd
import numpy as np

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
                 models_dict=None):
        
        super().__init__(pd_rules, index_cols=index_cols, categorical_cols=categorical_cols, drop_cols=drop_cols)

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
        '''Remove df as argument since don't want user to input any other df
        '''
        self.preprocess_prepare_for_clustering()

        cl = self.kmeans_optimal.predict(self.data_for_kmeans)

        cl_dict = dict(zip(self.data.index, cl))
        
        return cl_dict
