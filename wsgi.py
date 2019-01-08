import logging
logging.basicConfig(level=logging.INFO)

import os

from flask import Flask, jsonify, request
import requests

application = Flask(__name__)

@application.route("/", methods=['POST', 'PUT'])
def index():
    """Pass data to next endpoint."""
    data = request.get_json(force=True)

    application.logger.info(f'Received data {data}')

    # Identify itself
    data['ai_service'] = 'ai_clustering'

    # TODO: run inference for input data
    data['data'] = []

    host = os.environ.get('NEXT_MICROSERVICE_HOST')
    requests.post(f'http://{host}', json=data)

    application.logger.info('Done: %s', str(data))

    return jsonify(
        status='OK',
        message='Data processed by awesome and totally useful AI service'
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    application.run(host='0.0.0.0', port=port)
