import json
import os
import pickle
from flask import Flask
import clustering
import storage

app = Flask(__name__)

CLUSTERS = {date: storage.read(date) for date in set(storage.available())}

@app.route("/<date>")
def index(date):
    if date in CLUSTERS:
        return "<pre>%s</pre>" % CLUSTERS[date].to_string()
    else:
        return json.dumps(list(storage.available()))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

