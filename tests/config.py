class PreprocessSettings:
    index_cols = ['PassengerId', 'Name']
    categorical_cols = ['Sex', 'Embarked']
    drop_cols = ['Ticket', 'Cabin']

class KMeansSettings:
    n_clusters_low = 2
    n_clusters_high = 50
    n_clusters_stepsize = 1
    n_processes = None

class ModelNames:
    model_names = ['prepca_scaler', 'pca', 'postpca_scaler', 'kmeans']

class StorageSettings:
    type = "local"
    location = "models"
