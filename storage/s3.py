import os
import logging
import s3fs
import pyarrow.parquet as pq
import datetime
import pickle

fs = s3fs.S3FileSystem(secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                       key=os.environ.get("AWS_ACCESS_KEY_ID"))
SOURCE_BUCKET = os.environ.get("SOURCE_BUCKET", "insights-system-data")
DEST_BUCKET = os.environ.get("DEST_BUCKET", "aicoe-cluster-bucket")


def available(cache={}):
    try:
        if not cache:
            cache.update({os.path.basename(p).split(".")[0]: p for p in fs.ls(DEST_BUCKET)})
    except FileNotFoundError:
        cache = {}
    return cache


def unprocessed():
    processed = set(available())
    start = datetime.datetime(2018, 6, 1)
    while start < datetime.datetime.today():
        date = f"{start.year}-{start.month:02}-{start.day:02}"
        key = f"{SOURCE_BUCKET}/{date}/rule_data"
        if fs.exists(f"{key}/_SUCCESS") and date not in processed:
            logging.debug(f"Found {key}")
            yield key, date
        start += datetime.timedelta(days=1)


def get_dataset(key):
    return pq.ParquetDataset(key, filesystem=fs).read_pandas().to_pandas()


def write(data, date):
    path = f"{DEST_BUCKET}/{date}.p"
    with fs.open(path, "wb") as fp:
        logging.debug(f"Writing to {path}")
        pickle.dump(data, fp)


def read(date=None):
    if date in available():
        path = available()[date]
    else:
        raise FileNotFoundError

    with fs.open(path) as fp:
        logging.debug(f"Reading from {path}")
        return pickle.load(fp)
