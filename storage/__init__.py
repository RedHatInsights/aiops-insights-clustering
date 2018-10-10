import os
import logging


if "AWS_ACCESS_KEY_ID" in os.environ:
    logging.info("using AWS storage")
    from .s3 import *
elif "CEPH_SECRET" in os.environ:
    logging.info("using CEPH storage")
    from .ceph import *
else:
    logging.info("using LOCAL storage")
    from .local import *
