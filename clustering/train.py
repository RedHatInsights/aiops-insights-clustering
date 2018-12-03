import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.tree import DecisionTreeClassifier

import pickle

class Cluster:
	def __init__(self, pd_rules):
		self.data = pd_rules

		self.data_for_kmeans = None #zscore -> pca -> zscore

		self.n_clusters_low = n_clusters_low
		self.n_clusters_high = n_clusters_high
		self.n_clusters_stepsize = n_clusters_stepsize

		self.models_dict = None
		self.inertia_dict = None

		self.n_clusters_optimal = None
		self.kmeans_optimal = None

	def preprocess(self, df):
		df = self.preprocess_dropcols(df)
		df = self.preprocess_onehot(df)

		return df

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

		return df[r_cols].copy()

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
		#1-hot and drop columns
		self.data_preprocess = self.preprocess(self.data)

		#z-score -> pca -> z-score
		df_cluster_dict = self.preprocess_prepare_for_clustering(self, self.data_preprocess, pca_variance_threshold=0.99)		

		#df for clustering
		df_postpca_scaled = df_cluster_dict['df_postpca_scaled']

		self.data_for_kmeans = df_postpca_scaled

		self.models_dict, self.inertia_dict = self.find_nclusters(self, self.data_for_kmeans, self.n_clusters_low, self.n_clusters_high, self.n_clusters_stepsize):

		self.n_clusters_optimal = self.find_elbow(self.inertia_dict)

		self.kmeans_optimal = self.train_cluster(self.data_for_kmeans, self.n_clusters_optimal)

		#replace by ceph store
		self.save_model(df_cluster_dict['prepca_scaler'], 'prepca_scaler')
		self.save_model(df_cluster_dict['postpca_scaler'], 'postpca_scaler')
		self.save_model(df_cluster_dict['pca'], 'pca')

		self.save_model(self.kmeans_optimal, "kmeans_optimal")