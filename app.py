import logging
logging.basicConfig(level=logging.INFO)

import os
import clustering
import metric_tracking
import storage

def train(date):
    try:
        logging.info(f"Processing data for {date}")
        data = storage.get_dataset(date)

        # For testing: if you want to speed things up
        # data = data.sample(n=100)

        logging.info(f"Running clustering...")
        categorical_cols = None
        drop_cols = ["upload_time", "account", "system_id"]
        cluster = clustering.train.Cluster(
            data,
            categorical_cols=categorical_cols,
            drop_cols=drop_cols
        )
        cluster_info = cluster.train_the_cluster()

        logging.info(f"Writing results...")
        storage.write(cluster_info, date)
    except Exception:
        logging.exception(f"Failed to process data for {date}")


if __name__ == "__main__":
    if "AIOPS_TRAINING_DATE" in os.environ:
        train(os.getenv("AIOPS_TRAINING_DATE"))
        exit(0)

    if "MLFLOW_TRACKING_URI" in os.environ:
        logging.info("do tracking")
        metric_tracking.do_tracking()
        exit(0)
