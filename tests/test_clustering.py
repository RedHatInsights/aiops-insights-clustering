import pandas as pd
import sys
sys.path.append('../')

import storage
import clustering
import config

try:
    train_flag = int(sys.argv[1])
    inference_flag = int(sys.argv[2])
    print(train_flag, inference_flag)
except:
    raise ValueError("Usage: python test_clustering.py [train_flag] [inference_flag]")


def load_data():
    return pd.read_csv('test_dataset/train.csv')

data = load_data()

index_cols = config.PreprocessSettings.index_cols
categorical_cols = config.PreprocessSettings.categorical_cols
drop_cols = config.PreprocessSettings.drop_cols

n_clusters_low = config.KMeansSettings.n_clusters_low
n_clusters_high = config.KMeansSettings.n_clusters_high
n_clusters_stepsize = config.KMeansSettings.n_clusters_stepsize
n_processes = config.KMeansSettings.n_processes


if train_flag:
    print("Training...")
    #Train - instantiate object
    cluster = clustering.train.Cluster(data, index_cols=index_cols,
                                       categorical_cols=categorical_cols, drop_cols=drop_cols,
                                       n_clusters_low=n_clusters_low, n_clusters_high=n_clusters_high, 
                                       n_clusters_stepsize=n_clusters_stepsize, n_processes=n_processes)

    models_to_persist = cluster.train_the_cluster()

    storage.write(models_to_persist, '2018-12-12')

if inference_flag:
    print("Inference...")
    #Inference - instantiate object
    models_to_persist = storage.read('2018-12-12')

    inf = clustering.inference.Inference(data, index_cols=index_cols,
                              categorical_cols=categorical_cols, drop_cols=drop_cols,
                              models_dict=models_to_persist)
    
    #predict
    cl_dict = inf.predict()