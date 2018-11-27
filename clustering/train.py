import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering

import pickle

class Cluster:
	def __init__(self, pd_rules):
		self.data = pd_rules

	def preprocess(self):
		self.data = self.preprocess_dropcols(self.data)
		self.data = self.preprocess_onehot(self.data)

	def preprocess_onehot(self, df):
		'''one-host encoding
		'''
		df_list = []
		for col in df:
		    df_onehot = pd.get_dummies(df[col], prefix=col)
		    df_list.append(df_onehot)
		df_onehot = pd.concat(df_list, axis=1)		

		return df_onehot

	def preprocess_dropcols(self, df):
		'''Only keep rules columns
		Makes assumptions about data columns
		'''
		r_cols = [col for col in df_cluster.columns if col.find('r_')==0]
		c_cols = [col for col in df_cluster.columns if col.find('c_')==0]
		i_cols = [col for col in df_cluster.columns if col.find('i_')==0]

		assert len(r_cols)+len(c_cols)+len(i_cols)>0, "Input data does not contain columns starting with r_*, c_* and i_*"

		return df[r_cols]

	def preprocess_pca(self, df, pca_variance_threshold=0.99):
		pca = PCA()

		df_pca = pca.fit_transform(df)

		variance_cumsum = np.cumsum(pca.explained_variance_ratio_)

		gt_threshold = variance_cumsum[variance_cumsum > pca_variance_threshold]
		if len(gt_threshold):
			n_components = df.shape[1]
		else:
			n_components = np.where(variance_cumsum==gt_threshold[0])[0][0]

		return pca, df_pca, n_components

	def preprocess_normalize(self, df):
		scaler = StandardScaler()
		df_scaled = datascaler.fit_transform(df)

		return scaler, df_scaled

	def preprocess_prepare_for_clustering(self, df, pca_variance_threshold=0.99):
		scaler, df_scaled = self.preprocess_normalize(df)

		pca, df_pca, n_components = self.preprocess_pca(df_scaled, pca_variance_threshold=pca_variance_threshold)

		df_pca = df_pca.iloc[:,n_components]

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
		pass

	def train_cluster(self, df, n_clusters):
		model = KMeans(n_clusters=n_clusters)

		model.fit(df)

		return model

	def save_model(self, model, filename):
		pickle.dump(model, open(filename, 'wb'))

	def load_model(self, filename):
		return pickle.load(open(filename))		

