import logging

logging.basicConfig(level=logging.INFO)

import json
import os
from concurrent.futures import ProcessPoolExecutor
from flask import Flask
import storage
import sync
import metric_tracking

app = Flask(__name__)

CLUSTERS = {date: storage.read(date) for date in set(storage.available())}

@app.route("/<date>")
def index(date):
    if date in CLUSTERS:
        return json.dumps(CLUSTERS[date])
    else:
        return json.dumps(list(storage.available()))


@app.route("/sync")
def sync_endpoint():
    with ProcessPoolExecutor(max_workers=1) as executor:
        f = executor.submit(sync.sync)
        logging.info("Running sync job %s", f)
    return "OK"


if __name__ == "__main__":
    if "MLFLOW_TRACKING_URI" in os.environ:
        logging.info("syncing before metric tracking")
        sync.sync()
        logging.info("do tracking")
        metric_tracking.do_tracking()
        exit(0)

    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

