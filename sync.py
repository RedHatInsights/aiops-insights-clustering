import logging

logging.basicConfig(level=logging.DEBUG)

import clustering
import storage
import stability_score as ss

def sync():
    for key, date in storage.unprocessed():
        try:
            logging.info(f"Processing data for {date}")
            df = storage.get_dataset(key)
            logging.info(f"Running clustering...")
            cluster = clustering.cluster(df)
            logging.info(f"Writing results...")
            storage.write(cluster, date)
        except Exception:
            logging.exception(f"Failed to process data for {date}")

    logging.info(f"Scoring stability...")
    ss.run_clustering()


if __name__ == "__main__":
    sync()
