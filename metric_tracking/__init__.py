import logging
import storage
import mlflow

def do_tracking():
    available_dates = list(storage.available().keys())
    available_dates.sort()

    for date_a, date_b in zip(available_dates[:-1], available_dates[1:]):
        mlflow.start_run()
        mlflow.log_param("Date A", date_a)
        mlflow.log_param("Date B", date_b)
        calculate_score(date_a, date_b)
        mlflow.end_run()

def calculate_score(date_a, date_b):
    cluster_a = storage.read(date_a)
    cluster_b = storage.read(date_b)

    inverse_cluster_a = invert_cluster(cluster_a)
    inverse_cluster_b = invert_cluster(cluster_b)

    logging.info("calculate score for dates %s - %s" % (date_a, date_b))
    for cluster_id in inverse_cluster_a.keys():
        systems_a = frozenset(inverse_cluster_a[cluster_id])
        systems_b = frozenset(inverse_cluster_b[cluster_id])
        number_ids_in_both = len(systems_a.intersection(systems_b))

        median_number_ids_in_cluster = (len(systems_a) + len(systems_b)) / 2.0
        score = number_ids_in_both / median_number_ids_in_cluster

        mlflow.log_metric(("cluster_stability_%s" % cluster_id), score)
        logging.info("cluster_id %i - stability score %f" % (cluster_id, score))

def invert_cluster(cluster):
    new_cluster = {}
    for system_id, cluster_id in cluster.items():
        new_cluster[cluster_id] = new_cluster.get(cluster_id, [])
        new_cluster[cluster_id].append(system_id)
    return new_cluster

# def preprocess_rules_data(DF):
#
#     # Remove ID and time Columns from training set
#     DF = DF.iloc[:,1:-2]
#     # Convert any remaining textual data into a nan value (there are very few of these in the dataset)
#     DF = DF.apply(lambda L: pd.to_numeric(L,errors='coerce'))
#     # Fill in any remaning nan values with 0.
#     DF = DF.fillna(0)
#     return DF
#
#
# def group_clusters(results,num_clusters):
#
#     day = {}
#
#     # initialize our day dictionary with our initial classes
#     for i in range(num_clusters):
#         current_list = []
#         for j in range(len(results)):
#             if i == results[j][1]:
#                 current_list.append(results[j][0])
#         day[str(i)] = set(current_list)
#
#
#     return day
#
#
# def calculate_stability_score(day_1,day_2,num_clusters,ids_in_both_days):
#
#     stability_score = 0
#
#     for i in range(num_clusters):
#         smallest_difference = np.inf
#         for j in range(num_clusters):
#             #find best matching cluster:
#             current_difference  = len(day_1[str(i)].difference(day_2[str(j)]))
#             if current_difference < smallest_difference:
#                 smallest_difference = current_difference
#         #print(i, " : ", smallest_difference)
#         stability_score += smallest_difference
#
#     stability_score = ((1-(stability_score/len(ids_in_both_days))) * 100.0 )
#     #print(str(stability_score) , "%,")
#
#     return stability_score
#
#
# def run_clustering():
#
#
#     ## Read in our environment variables for storage, tracking, and model ##
#
#     # Storage Parameters
#     ceph_key = os.environ.get("CEPH_KEY")
#     ceph_secret = os.environ.get("CEPH_SECRET")
#     ceph_host = os.environ.get("CEPH_HOST")
#     ceph_bucket = os.environ.get("CEPH_BUCKET")
#
#     # MLFlow Paramters
#     mlflow_experiment_id = os.environ.get('MLFLOW_EXPERIMENT_ID')
#     mlflow_tracking_uri  = os.environ.get("MLFLOW_TRACKING_URI")
#
#
#     # Model Parameters
#     k_clusters = int(os.environ.get("K_CLUSTERS"))
#     pca_dimensions = int(os.environ.get("PCA_DIMENSIONS"))
#     date_1 = os.environ.get("DAY_1")
#     date_2 = os.environ.get("DAY_2")
#
#     # Parser to be used. This can be a configurable paramter in the futre, but for now we are focused only on this dataset.
#     parser = "rule_data"
#
#
#     # Set up connection to our CEPH storage
#     client_kwargs = { 'endpoint_url' : ceph_host }
#     s3 = s3fs.S3FileSystem(secret=ceph_secret, key=ceph_key, client_kwargs=client_kwargs)
#
#
#
#     ## DAY 1 READ IN AND PREPROCESS ##
#
#     # Read in rules data for the first day
#     url = "DH-DEV-INSIGHTS/{}/{}".format(date_1, parser)
#     rulesx_day_1 = pq.ParquetDataset(url,filesystem=s3).read_pandas().to_pandas()
#     print(len(rulesx_day_1),": data points from day 1")
#
#     # Preprocess the dataset
#     rules_day_1 = preprocess_rules_data(rulesx_day_1)
#
#
#     # normalize the data prior to PCA
#     min_max_scaler_1 = preprocessing.StandardScaler()
#     np_scaled_1 = min_max_scaler_1.fit_transform(rules_day_1)
#     data_1 = pd.DataFrame(np_scaled_1)
#
#     # Reduce the dimensionality of the data with PCA
#     pca_1 = PCA(n_components=pca_dimensions)
#     data_transformed_1 = pca_1.fit_transform(data_1)
#
#
#     ## DAY 2 READ IN AND PREPROCESS ##
#
#     # Read in rules data for the first day
#     url = "DH-DEV-INSIGHTS/{}/{}".format(date_2, parser)
#     rulesx_day_2 = pq.ParquetDataset(url,filesystem=s3).read_pandas().to_pandas()
#     print(len(rulesx_day_2),": data points from day 2")
#
#     # Preprocess the dataset
#     rules_day_2 = preprocess_rules_data(rulesx_day_2)
#
#
#     # normalize the data prior to PCA
#     min_max_scaler_2 = preprocessing.StandardScaler()
#     np_scaled_2 = min_max_scaler_2.fit_transform(rules_day_2)
#     data_2 = pd.DataFrame(np_scaled_2)
#
#     # Reduce the dimensionality of the data with PCA
#     pca_2 = PCA(n_components=pca_dimensions)
#     data_transformed_2 = pca_2.fit_transform(data_2)
#
#
#
#     ## Perform the K-means clustering for both days and compare results ##
#
#     # probably over-kill here, but just to ensure we are tracking in the correct location, we re-define these variables.
#     mlflow.set_tracking_uri("http://mlflow-server-svc-vpavlin-jupyterhub.cloud.paas.upshift.redhat.com")
#     os.environ["MLFLOW_EXPERIMENT_ID"]='2'
#
#     # Start the mlflow tracking
#     with mlflow.start_run():
#
#
#         # Run Kmeans on our dataset for day 1 and collect the lables
#         kmeans_1 = KMeans(n_clusters=k_clusters).fit(data_transformed_1)
#         labels_1 = kmeans_1.labels_
#         centroids_1 = kmeans_1.cluster_centers_
#
#         # create a list of the system ids associates with the first day's data set
#         sysids_1 = rulesx_day_1.iloc[:,-2]
#
#         # Collect the id's and the results into a numpy array called results_1
#         results_1 = np.vstack((np.array(sysids_1),labels_1)).T
#
#         # Run Kmeans on our day 2 dataset and collect the labels
#         kmeans_2 = KMeans(n_clusters=k_clusters).fit(data_transformed_2)
#         labels_2 = kmeans_2.labels_
#         centroids_2 = kmeans_2.cluster_centers_
#
#         # Create a list od the system ids associated with the second day's data set
#         sysids_2 = rulesx_day_2.iloc[:,-2]
#
#         # Collect the id's and the results into a numpy array called results_2
#         results_2 = np.vstack((np.array(sysids_2),labels_2)).T
#
#         # Convert both lists of system ids into sets for a quick comparisons
#         one = set(list(sysids_1))
#         two = set(list(sysids_2))
#
#         # Create a set of only those days that are in both day_1 and day_2 as they are the only ones relevent for calculating our metric.
#         ids_in_both_days = one.intersection(two)
#         print(len(ids_in_both_days), "in both days")
#
#         # Clear Results_1 of id's that are not present in day 2
#         new_results = np.in1d(results_1[:,0],np.array(list(ids_in_both_days)))
#         x = results_1[new_results]
#
#         # Create a dictionary of all id's clustered together {Cluster label: Set of system ids}
#         day_1 = group_clusters(x,k_clusters)
#
#         # Clear Results_2 of id's that are not present in day 1.
#         new_results = np.in1d(results_2[:,0],np.array(list(ids_in_both_days)))
#         x = results_2[new_results]
#
#         # Create a dictonary of all id's clustered together {Cluster label: Set of system ids}
#         day_2 = group_clusters(x,k_clusters)
#
#         # Calculate the similarity between the two days.
#         stability_score = calculate_stability_score(day_1, day_2, k_clusters, ids_in_both_days)
#
#         mlflow.log_param("K-Clusters", k_clusters )
#         mlflow.log_param("PCA_Dimensions", pca_dimensions)
#         mlflow.log_param("Shared_ids_between_days", len(ids_in_both_days))
#         mlflow.log_param("Date 1", date_1)
#         mlflow.log_param("Date 2", date_2)
#         mlflow.log_metric("cluster_stability", stability_score)
#
#         # Store results in Ceph
#         # Need centroids (ndarray) to reinitialize K-means object
#         data = {'day1': {'labels': labels_1, 'centroids': centroids_1, 'results': results_1},
#                 'day2': {'labels': labels_2, 'centroids': centroids_2, 'results': results_2},
#                 'stability': stability_score}
#         ceph.write(data)
