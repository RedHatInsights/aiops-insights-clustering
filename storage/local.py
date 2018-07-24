import pickle
import os
import glob
import pyarrow.parquet as pq
import logging
import datetime

def available(cache={}):
    try:
        if not cache:
            cache.update({os.path.basename(p).split(".")[0]: p for p in glob.glob('data/*.p')})
    except FileNotFoundError:
        cache = {}
    return cache


def unprocessed():
    processed = set(available())
    start = datetime.datetime(2018, 6, 1)
    while start < datetime.datetime.today():
        date = f"{start.year}-{start.month:02}-{start.day:02}"
        key = f"data/{date}/rule_data"
        if os.path.exists(f"{key}/_SUCCESS") and date not in processed:
            logging.debug(f"Found {key}")
            yield key, date
        start += datetime.timedelta(days=1)


def get_dataset(key):
    return pq.ParquetDataset(key).read_pandas().to_pandas()


def write(data, date):
    with open(f"data/{date}.p", "wb") as fp:
        pickle.dump(data, fp)


def read(date=None):
    if date in available():
        path = available()[date]
    else:
        raise FileNotFoundError

    with open(path, 'rb') as fp:
        logging.debug(f"Reading from {path}")
        return pickle.load(fp)
