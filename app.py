import os
import pickle
import clustering
from flask import Flask

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

