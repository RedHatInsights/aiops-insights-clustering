import train
import inference
import pandas as pd
import os
import sys

try:
	train_flag = int(sys.argv[1])
	inference_flag = int(sys.argv[2])
	print(train_flag, inference_flag)
except:
	raise ValueError("Usage: python test.py [train_flag] [inference_flag]")

'''Notes:
The API is inconsistent in places. For example, n_clusters_X are both passed to the constructor
but also passed explicitly to find_nclusters. Will fix issues like these in the next iteration.
'''

def load_data():
	return pd.read_csv('test_dataset/train.csv')

data = load_data()
categorical_cols = ['Sex', 'Embarked']
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']


if train_flag:
	print("Training...")
	#Train - instantiate object
	cluster = train.Cluster(data,
							n_clusters_low = 2,
							n_clusters_high = 10,
							n_clusters_stepsize = 1)

	#Train - preprocess = one-hot encoding of categorical cols, drop columns, and fill nulls with 0s (for now)
	cluster.data_preprocess = cluster.preprocess(cluster.data, categorical_cols=categorical_cols, drop_cols=drop_cols)

	#Train - prepare for clustering (z-score -> PCA -> z-score)
	df_cluster_dict = cluster.preprocess_prepare_for_clustering(cluster.data_preprocess)
	#df_cluster_dict contains 3 models (two scalers and 1 PCA) and 3 dataframes (after each step)

	#scan n_clusters and find optimal point (In progress)
	cluster.data_for_kmeans = df_cluster_dict['df_postpca_scaled']
	cluster.models_dict, cluster.inertia_dict = cluster.find_nclusters(cluster.data_for_kmeans, cluster.n_clusters_low, cluster.n_clusters_high, cluster.n_clusters_stepsize)
	cluster.n_clusters_optimal = cluster.find_elbow(cluster.inertia_dict) #hard-coded currently

	cluster.kmeans_optimal = cluster.train_cluster(cluster.data_for_kmeans, cluster.n_clusters_optimal)

	#need to wrap in a function
	model_loc = 'models'
	if not os.path.exists(model_loc):
		os.mkdir(model_loc)
	cluster.save_model(df_cluster_dict['prepca_scaler'], f'{model_loc}/prepca_scaler.pkl')
	cluster.save_model(df_cluster_dict['postpca_scaler'], f'{model_loc}/postpca_scaler.pkl')
	cluster.save_model(df_cluster_dict['pca'], f'{model_loc}/pca.pkl')
	cluster.save_model(cluster.kmeans_optimal, f'{model_loc}/kmeans.pkl')

if inference_flag:
	print("Inference...")
	#Inference - instantiate object
	inf = inference.Inference(data,
					categorical_cols=categorical_cols,
					drop_cols=drop_cols)
	
	#load pre-trained models (need to wrap in a function)
	model_loc = 'models'
	prepca_scaler_filename = f'{model_loc}/prepca_scaler.pkl'
	postpca_scaler_filename = f'{model_loc}/postpca_scaler.pkl'
	pca_filename = f'{model_loc}/pca.pkl'
	cluster_filename = f'{model_loc}/kmeans.pkl'
	inf.load_models(prepca_scaler_filename, 
					pca_filename, 
					postpca_scaler_filename,
					cluster_filename)

	#prepare data for clustering
	inf.preprocess_prepare_for_clustering()

	#predict
	cl_dict = inf.predict()