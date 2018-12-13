import pandas as pd
import clustering
import clustering.config as config

data = pd.read_csv('tests/test_dataset/train.csv')
model = {}

def test_training():
    global model, data
    index_cols = config.PreprocessSettings.index_cols
    categorical_cols = config.PreprocessSettings.categorical_cols
    drop_cols = config.PreprocessSettings.drop_cols
    n_clusters_low = config.KMeansSettings.n_clusters_low
    n_clusters_high = config.KMeansSettings.n_clusters_high
    n_clusters_stepsize = config.KMeansSettings.n_clusters_stepsize
    n_processes = config.KMeansSettings.n_processes

    cluster = clustering.train.Cluster(data, index_cols=index_cols,
                                       categorical_cols=categorical_cols, drop_cols=drop_cols,
                                       n_clusters_low=n_clusters_low, n_clusters_high=n_clusters_high,
                                       n_clusters_stepsize=n_clusters_stepsize, n_processes=n_processes)

    model = cluster.train_the_cluster()

    assert model["prepca_scaler"]

def test_inference():
    global model, data

    index_cols = config.PreprocessSettings.index_cols
    categorical_cols = config.PreprocessSettings.categorical_cols
    drop_cols = config.PreprocessSettings.drop_cols

    inf = clustering.inference.Inference(data, index_cols=index_cols,
                              categorical_cols=categorical_cols, drop_cols=drop_cols,
                              models_dict=model)

    cl_dict = inf.predict()
    assert len(cl_dict) == 891
