import pandas as pd
import numpy as np

import train

'''
Save n_components in pca object and get rid of get_pca_ncomponents
Add tests:
	???
Ensure any df going to predict goes through preprocess
'''

class Inference(train.Cluster):
	def __init__(self, pd_rules, categorical_cols=None, drop_cols=None):
		super().__init__(pd_rules)

		self.data_preprocess = self.preprocess(pd_rules, categorical_cols=categorical_cols, drop_cols=drop_cols)

	def load_models(self, prepca_scaler_filename, pca_filename, postpca_scaler_filename, cluster_filename):
		self.prepca_scaler = self.load_model(prepca_scaler_filename)
		self.pca = self.load_model(pca_filename)
		self.postpca_scaler = self.load_model(postpca_scaler_filename)
		self.kmeans_optimal = self.load_model(cluster_filename)

	def get_pca_ncomponents(self, pca, pca_variance_threshold):
		variance_cumsum = np.cumsum(pca.explained_variance_ratio_)

		gt_threshold = variance_cumsum[variance_cumsum > pca_variance_threshold]
		if len(gt_threshold)==0:
			n_components = len(pca.explained_variance_ratio_)
		else:
			n_components = np.where(variance_cumsum==gt_threshold[0])[0][0]

		return n_components

	def preprocess_prepare_for_clustering(self):
		df_prepca_scaled = self.prepca_scaler.transform(self.data_preprocess)		
		
		df_pca = pd.DataFrame(self.pca.transform(df_prepca_scaled))
		df_pca = df_pca.iloc[:,:self.pca.n_components]

		df_postpca_scaled = self.postpca_scaler.transform(df_pca)

		self.data_for_kmeans = df_postpca_scaled

	def predict(self):		
		'''Remove df as argument since don't want user to input any other df
		'''
		cl = self.kmeans_optimal.predict(self.data_for_kmeans)

		return cl
