import logging
import os
import warnings

from flask import Flask, jsonify, request
from flask.logging import default_handler
from werkzeug.exceptions import BadRequest
from sklearn.exceptions import DataConversionWarning

from workers import prediction_worker


def create_application():
    """Create Flask application instance with AWS client enabled."""
    app = Flask(__name__)
    app.config['NEXT_MICROSERVICE_HOST'] = \
        os.environ.get('NEXT_MICROSERVICE_HOST')

    return app


APP = create_application()
ROOT_LOGGER = logging.getLogger()
ROOT_LOGGER.setLevel(APP.logger.level)
ROOT_LOGGER.addHandler(default_handler)

# Disable sklearn implicit data conversion warnings to clean-up server logs
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


@APP.route("/", methods=['POST', 'PUT'])
def index():
    """Pass data to next endpoint."""
    next_service = APP.config['NEXT_MICROSERVICE_HOST']

    try:
        raw_data = request.get_data()
        job_id = request.headers['source_id']
    except (BadRequest, KeyError):
        return jsonify(
            status='ERROR',
            message="Unable to parse input data JSON."
        ), 400

    b64_identity = request.headers.get('x-rh-identity')

    prediction_worker(job_id, raw_data, next_service, b64_identity)
    APP.logger.info('Job started')

    return jsonify(
        status='OK',
        message='Clustering initiated.'
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8004))
    APP.run(port=port)
