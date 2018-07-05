import os


if "AWS_ACCESS_KEY_ID" in os.environ:
    from .s3 import *
elif "CEPH_SECRET" in os.environ:
    from .ceph import *
else:
    from .local import *
