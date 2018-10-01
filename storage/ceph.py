import pyarrow.parquet as pq
import s3fs, datetime, os

(ceph_key, ceph_secret, ceph_endpoint) = (os.getenv('CEPH_KEY'), os.getenv('CEPH_SECRET'), os.getenv('CEPH_ENDPOINT'))
client_kwargs = {'endpoint_url': ceph_endpoint}


def get_dataset(key):
    filesystem = s3fs.S3FileSystem(secret=ceph_secret, key=ceph_key, client_kwargs=client_kwargs)
    return pq.ParquetDataset(key, filesystem=filesystem).read_pandas().to_pandas()


def write(data):
	date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	write(data, date)


def read(date):
	return read(date=date)