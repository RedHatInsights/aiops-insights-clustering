import logging
import os

import requests
from flask import Flask, jsonify, request
from flask.logging import default_handler
from flask.exception import BadRequest

from workers import QUEUE, init_workers


application = Flask(__name__)

# Sync logging between Flask and Gunicorn
gunicorn_logger = logging.getLogger('gunicorn.error')
application.logger.handlers = gunicorn_logger.handlers
application.logger.setLevel(gunicorn_logger.level)

# Check presence of next endpoint
assert os.environ.get('NEXT_MICROSERVICE_HOST')

@application.before_first_request
def startup():
    count = os.environ.get('WORKERS_COUNT', 1)
    application.logger.info("Starting workers (%d)", count)
    init_workers(count)


@application.route("/", methods=['POST', 'PUT'])
def index():
    """Pass data to next endpoint."""
    try:
        input_data = request.get_json(force=True, cache=False)
    except BadRequest:
        return jsonify(
            status='ERROR',
            message="Unable to parse input data JSON."
        ), 400

    application.logger.info(f'Received a Job...')
    QUEUE.put(input_data)
    application.logger.info('Queued Job ID: %s', data.get('id'))

    return jsonify(
        status='OK',
        message='Clustering initiated.'
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    application.run(host='0.0.0.0', port=port)
