import logging
import os

from flask import Flask

import clustering
import metric_tracking
import storage

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)


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
        logging.info("syncing before metric tracking")
        # sync.sync()
        logging.info("do tracking")
        metric_tracking.do_tracking()
        exit(0)

    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
