import pickle
import pyarrow.parquet as pq


def get_dataset(key):
    return pq.ParquetDataset(key).read_pandas().to_pandas()


def write(data, date):
    with open(f"{date}.p", "wb") as fp:
        pickle.dump(data, fp)
