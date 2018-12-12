import datetime
import logging
import os
import pickle
import re

import pyarrow.parquet as pq
import s3fs

(ceph_key, ceph_secret, ceph_endpoint) = (os.getenv('CEPH_KEY'), os.getenv('CEPH_SECRET'), os.getenv('CEPH_ENDPOINT'))
SOURCE_BUCKET = os.getenv('CEPH_BUCKET')
DEST_BUCKET = SOURCE_BUCKET + '/insights-clustering'
client_kwargs = {'endpoint_url': ceph_endpoint}

fs = s3fs.S3FileSystem(secret=ceph_secret, key=ceph_key, client_kwargs=client_kwargs)


def available(cache={}):
    if cache:
        return cache

    try:
        for p in fs.ls(DEST_BUCKET):
            date = os.path.basename(p).split(".")[0]
            if re.match(r"\d{4}-\d{2}-\d{2}", date):
                cache.update({date: p})
    except FileNotFoundError:
        cache = {}

    return cache


def unprocessed():
    processed = set(available())
    start = datetime.datetime(2018, 6, 1)
    while start < datetime.datetime.today():
        date = start.strftime("%y-%m-%d")
        key = f"{SOURCE_BUCKET}/{date}/rule_data"
        if fs.exists(f"{key}/_SUCCESS") and date not in processed:
            logging.debug(f"Found {key}")
            yield key, date
        start += datetime.timedelta(days=1)


def get_dataset(date):
    key = f"{SOURCE_BUCKET}/{date}/rule_data"
    return pq.ParquetDataset(key, filesystem=fs).read_pandas().to_pandas()


def write(data, date):
    path = f"{DEST_BUCKET}/{date}.p"
    with fs.open(path, "wb") as fp:
        logging.debug(f"Writing to {path}")
        pickle.dump(data, fp)


def read(date):
    all_clusters = available()
    if date in all_clusters:
        path = all_clusters[date]
    else:
        raise FileNotFoundError

    with fs.open(path) as fp:
        logging.debug(f"Reading from {path}")
        return pickle.load(fp)
