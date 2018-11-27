import pandas as pd
import numpy as np

import train

class Inference(Train):
	def __init__(self, pd_rules):
		super().__init__(pd_rules)

	def predict(self, df, prepca_scaler_filename, pca_filename, postpca_scaler_filename, cluster_filename):
		prepca_scaler = self.load_model(prepca_scaler_filename)
		pca = self.load_model(pca_filename)
		postpca_scaler = self.load_model(postpca_scaler_filename)
		cluster = self.load_model(cluster_filename)

		df_scaled = prepca_scaler.transform(df)
		df_pca = pca.transform(df_scaled)
		df_pca_scaled = postpca_scaler.transform(df_pca)

		cl = cluster.predict(df_pca_scaled)

		return cl
