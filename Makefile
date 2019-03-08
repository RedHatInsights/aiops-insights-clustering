ENV_FILE := .env
include ${ENV_FILE}
export $(shell sed 's/=.*//' ${ENV_FILE})
export PIPENV_DOTENV_LOCATION=${ENV_FILE}

oc_build_head:
	git archive --format=tar.gz HEAD > build/HEAD.tar.gz
	oc start-build systems-clustering --from-archive build/HEAD.tar.gz --follow

oc_cluster_train:
	oc new-app --template systems-clustering-job \
		--param AIOPS_TRAINING_DATE=${AIOPS_TRAINING_DATE} \
		--param CEPH_KEY=${CEPH_KEY} \
		--param CEPH_SECRET=${CEPH_SECRET} \
		--param CEPH_ENDPOINT=${CEPH_ENDPOINT} \
		--param CEPH_BUCKET=${CEPH_BUCKET}

oc_run_experiment:
	oc new-app mlflow-experiment-job --param APP_IMAGE_URI=your-application-image-name\
		--param LIMIT_CPU=4 \
		--param LIMIT_MEM=16G \
                --env AIOPS_TRAINING_DATE=${AIOPS_TRAINING_DATE} \
                --env CEPH_KEY=${CEPH_KEY} \
                --env CEPH_SECRET=${CEPH_SECRET} \
                --env CEPH_ENDPOINT=${CEPH_ENDPOINT} \
                --env CEPH_BUCKET=${CEPH_BUCKET}

test:
	pipenv run pytest
