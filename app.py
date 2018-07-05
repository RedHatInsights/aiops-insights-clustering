import os
import pickle
import clustering
from flask import Flask
import storage
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
cluster = None
if os.path.isfile('cluster.p'):
    cluster = pickle.load(open('cluster.p', 'rb'))


@app.route("/")
def index():
    global cluster

    if cluster is not None:
        return "<pre>%s</pre>" % cluster.to_string()
    else:
        return "hit /update to create the clustering"

@app.route("/update")
def update():
    global cluster

    cluster = clustering.cluster()
    pickle.dump(cluster, open('cluster.p', 'wb'))

    return "Clustering updated"


def sync():
    for key, date in storage.unprocessed():
        try:
            logging.info(f"Processing data for {date}")
            df = storage.get_dataset(key)
            logging.info(f"Running clustering...")
            cluster = clustring.cluster(df)
            logging.info(f"Writing results...")
            storage.write(cluster, date)
        except Exception:
            logging.exception(f"Failed to process data for {date}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

