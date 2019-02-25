import logging
from threading import Thread, current_thread
from io import BytesIO

import requests
import pandas as pd

import clustering
# Take the example config for now
import tests.config as config


logger = logging.getLogger()
MAX_RETRIES = 3


def _retryable(method: str, *args, **kwargs) -> requests.Response:
    """Retryable HTTP request.

    Invoke a "method" on "requests.session" with retry logic.
    :param method: "get", "post" etc.
    :param *args: Args for requests (first should be an URL, etc.)
    :param **kwargs: Kwargs for requests
    :return: Response object
    :raises: HTTPError when all requests fail
    """
    thread = current_thread()

    with requests.Session() as session:
        for attempt in range(MAX_RETRIES):
            try:
                resp = getattr(session, method)(*args, **kwargs)

                resp.raise_for_status()
            except (requests.HTTPError, requests.ConnectionError) as e:
                logger.warning(
                    '%s: Request failed (attempt #%d), retrying: %s',
                    thread.name, attempt, str(e)
                )
                continue
            else:
                return resp

    raise requests.HTTPError('All attempts failed')


def _train(data: pd.DataFrame) -> dict:
    """Create and train model.

    Use the Cluster definition from clustering package and train in on the
    data.
    :param data: Training data set
    """
    cluster = clustering.train.Cluster(
        data,
        index_cols=config.PreprocessSettings.index_cols,
        categorical_cols=config.PreprocessSettings.categorical_cols,
        drop_cols=config.PreprocessSettings.drop_cols,
        n_clusters_low=config.KMeansSettings.n_clusters_low,
        n_clusters_high=config.KMeansSettings.n_clusters_high,
        n_clusters_stepsize=config.KMeansSettings.n_clusters_stepsize,
        n_processes=config.KMeansSettings.n_processes
    )

    return cluster.train_the_cluster()


def _inference(model: dict, data: pd.DataFrame) -> dict:
    """Predict clusters.

    Use give model to predict clusters for data on input."""
    inf = clustering.inference.Inference(
        data,
        index_cols=config.PreprocessSettings.index_cols,
        categorical_cols=config.PreprocessSettings.categorical_cols,
        drop_cols=config.PreprocessSettings.drop_cols,
        models_dict=model
    )

    return inf.predict()


def prediction_worker(id: str,
                      raw_data: bytes,
                      next_service: str,
                      b64_identity: str = None) -> Thread:
    def worker() -> None:
        thread = current_thread()
        logger.debug('%s: Worker started', thread.name)

        if not all((id, raw_data)):
            logger.error("%s: Invalid Job data, terminated.", thread.name)
            return

        logger.info('%s: Job ID %s: Started...', thread.name, id)
        try:
            data = pd.read_csv(BytesIO(raw_data))
        except ValueError:
            logger.error(
                "%s: Job ID %s: Unable to parse data, terminated.",
                thread.name, id
            )
            return

        logger.info(
            '%s: Job ID %s: Training model and predicting clusters...',
            thread.name, id
        )
        try:
            model = _train(data)
            clusters = _inference(model, data)
        except KeyError as e:
            logger.error(
                "%s: Job ID %s: Preprocessing failed: Missing fields %s",
                thread.name, id, e
            )
            return

        # Build response JSON
        output = {
            'id': id,
            'ai_service': 'ai_clustering',
            'data': clusters
        }

        logger.info(
            '%s: Job ID %s: Prediction done, publishing...',
            thread.name, id
        )
        # Pass to the next service
        try:
            _retryable(
                'post',
                f'http://{next_service}',
                json=output,
                headers={"x-rh-identity": b64_identity}
            )
        except requests.HTTPError as exception:
            logger.error(
                '%s: Failed to pass data for "%s": %s',
                thread.name, id, exception
            )

        logger.debug('%s: Done, exiting', thread.name)

    thread = Thread(target=worker)
    thread.start()

    return thread
