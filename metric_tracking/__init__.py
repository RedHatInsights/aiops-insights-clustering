import logging
import storage

import mlflow

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

        mlflow.log_metric(("cluster_stability_%s" % cluster_id), score)
        logging.info("cluster_id %i - stability score %f" % (cluster_id, score))

def invert_cluster(cluster):
    new_cluster = {}
    for system_id, cluster_id in cluster.items():
        new_cluster[cluster_id] = new_cluster.get(cluster_id, [])
        new_cluster[cluster_id].append(system_id)
    return new_cluster

