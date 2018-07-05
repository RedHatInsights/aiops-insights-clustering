import os
import s3fs
import pyarrow.parquet as pq
import datetime
import pickle

fs = s3fs.S3FileSystem(secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                       key=os.environ.get("AWS_ACCESS_KEY_ID"))
SOURCE_BUCKET = os.environ.get("SOURCE_BUCKET", "insights-system-data")
DEST_BUCKET = os.environ.get("DEST_BUCKET", "aicoe-cluster-data")


def get_clusters():
    try:
        return {os.path.basename(p).split(".")[0]: p for p in fs.ls(DEST_BUCKET)}
    except FileNotFoundError:
        return {}


def unprocessed():
    processed = set(get_clusters())
    start = datetime.datetime(2018, 5, 1)
    while start < datetime.datetime.today():
        date = f"{start.year}-{start.month:02}-{start.day:02}"
        key = f"{SOURCE_BUCKET}/{date}/rule_data"
        if fs.exists(f"{key}/_SUCCESS") and date not in processed:
            yield key, date
        start += datetime.timedelta(days=1)


def get_dataset(key):
    return pq.ParquetDataset(key, filesystem=fs).read_pandas().to_pandas()


def write(data, date):
    path = f"{DEST_BUCKET}/{date}.p"
    with fs.open(path) as fp:
        pickle.dump(data, fp)


def read(date=None):
    available = get_clusters()
    if date and date in available:
        path = available[date]
    else:
        path = max(available.items())[1]

    with fs.open(path) as fp:
        return pickle.load(fp)
