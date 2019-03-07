import logging
import storage

import mlflow
from sklearn import metrics

# for a more sophisticated method have a look at
#  https://gitlab.cee.redhat.com/mcliffor/insights_clustering/blob/master/stability_score.py

def do_tracking():
    available_dates = list(storage.available().keys())
    available_dates.sort()

    for date_a, date_b in zip(available_dates[:-1], available_dates[1:]):
        mlflow.start_run()
        mlflow.log_param("Date A", date_a)
        mlflow.log_param("Date B", date_b)
        calculate_score(date_a, date_b)
        calculate_metrics(date_a,date_b)
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
        # So now if you have 100 system ids in first and 160 in second
        # (100 are the same and we have 60 new), you'll count the score as
        # 100 / (100+160/2) = 100 / 130 = 0.76.
        # While it should be 1, if the 100 systems were clustered the same.
        # We can't say anything about the 60 new systems (unless we do other analysis of them),
        # so those needs to be ignored.
        # TODO: only look at systems present in both runs

        number_ids_in_both = len(systems_a.intersection(systems_b))

        median_number_ids_in_cluster = (len(systems_a) + len(systems_b)) / 2.0
        score = number_ids_in_both / median_number_ids_in_cluster
        
        mlflow.log_metric(("number_of_ids_in_both_days for cluster_%s" cluster_id),number_ids_in_both)
        mlflow.log_metric(("cluster_stability_%s" % cluster_id), score)
        logging.info("cluster_id %i - stability score %f" % (cluster_id, score))

def invert_cluster(cluster):
    new_cluster = {}
    for system_id, cluster_id in cluster.items():
        new_cluster[cluster_id] = new_cluster.get(cluster_id, [])
        new_cluster[cluster_id].append(system_id)
    return new_cluster

def calculate_metrics(date_a,date_b):
    cluster_a = storage.read(date_a)
    cluster_b = storage.read(date_b)
    
    #Find the list of common system ids in both the days
    sys_ids_a = find_sys_ids(cluster_a)
    sys_ids_b = find_sys_ids(cluster_b)
    sys_ids_in_both = list(set(sys_ids_a).intersection(set(sys_ids_b)))
    num_sys_ids_in_both = len(set(sys_ids_a).intersection(set(sys_ids_b)))
    
    #Find the cluster labels/ids for the common system ids in both days
    labels_a, labels_b = find_labels(sys_ids_in_both,cluster_a,cluster_b)
    
    #Find the performance metrics such as mutual information score, fowlkes mallows score
    mutual_info_score = metrics.mutual_info_score(labels_a, labels_b)
    fowlkes_mallows_score = metrics.fowlkes_mallows_score(labels_a, labels_b)
    adjusted_rand_score = metrics.adjusted_rand_score(labels_a, labels_b)
    
    mlflow.log_metric("total_number_of_ids_in_both_days",num_sys_ids_in_both)
    mlflow.log_metric("mutual_info_score",mutual_info_score)
    mlflow.log_metric("fowlkes_mallows_score",fowlkes_mallows_score)
    mlflow.log_metric("adjusted_rand_score",adjusted_rand_score)
    
def find_sys_ids(cluster):
    sys_ids = sorted(list(cluster.keys())
    return sys_ids

def find_new_labels(common_sys_ids,cluster_1,cluster_2):
    cluster_labels_1 = []
    cluster_labels_2 = []
    for system_id in sorted(common_sys_ids):
        cluster_labels_1.append(cluster_1[system_id])
        cluster_labels_2.append(cluster_2[system_id])
    return cluster_labels_1, cluster_labels_2
