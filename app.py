import logging

logging.basicConfig(level=logging.INFO)

import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from flask import Flask
import clustering
import storage
import sync

app = Flask(__name__)

CLUSTERS = {date: storage.read(date) for date in set(storage.available())}

@app.route("/<date>")
def index(date):
    if date in CLUSTERS:
        return "<pre>%s</pre>" % CLUSTERS[date].to_string()
    else:
        return json.dumps(list(storage.available()))


@app.route("/sync")
def sync_endpoint():
    with ProcessPoolExecutor(max_workers=1) as executor:
        f = executor.submit(sync.sync)
        logging.info("Running sync job %s", f)
    return "OK"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

