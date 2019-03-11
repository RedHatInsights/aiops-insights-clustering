# Clustering Systems

## Running the clustering on OpenShift

First you'll load the template that has all required resources

```
❯ oc create -f ./cluster-job-template.yaml -f ./build-config-template.yaml
template "systems-clustering-job" created
template "systems-clustering-bc-is" created
```

Then create the BuildConfig and ImageStream

```
❯ oc new-app --template systems-clustering-bc-is
--> Deploying template "mhild-test/systems-clustering-bc-is" to project mhild-test

     * With parameters:
        * APPLICATION_NAME=systems-clustering
        * GIT_URI=https://github.com/ManageIQ/aiops-insights-clustering.git

--> Creating resources ...
    buildconfig "systems-clustering" created
    imagestream "systems-clustering" created
--> Success
    Use 'oc start-build systems-clustering' to start a build.
    Run 'oc status' to view your app.
```

## Development workflow

Copy .env file and adjust variables

```
cp .env.example .env
```

Start a build

```
❯ oc start-build systems-clustering
build "systems-clustering-1" started
```

Or to push your local committed code

```
g add .
make oc_build_head
```

And finally you can run a job that does the clustering

```
make oc_cluster_train
```

Look at the output of the job

```
❯ oc logs systems-clustering-job-spah -f
---> Running application from Python script (app.py) ...
app.py:50: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
  np_rules = pd_rules.as_matrix()
0        1
1        4
2        1
        ..
40160    3
40161    1
Name: cluster, Length: 40162, dtype: int32
```

## Tests

```
make test
```

## Running the MLflow experiment on OpenShift

This is a two stage process:
- The first stage is to get the application code on OpenShift.
- The second stage is to run the experiment

Once you have the code on openshift, you can have multiple instances of your experiment running with different parameters.

## First Stage: Obtaining the application code on OpenShift

To run this application on openshift, we need to create a container image for it. In other words, this means that we need to download the code that we want to run on openshift. To do this a image build template should already be available in your openshift namespace.

To download the source code for our experiment to openshift, use the following command:

```
    oc process mlflow-experiment-bc --param APPLICATION_NAME=aiops-insights-clustering --param GIT_URI=https://github.com/ManageIQ/aiops-insights-clustering.git | oc create -f -

```
Building the container image can take a couple of minutes or more depending on the number of packages that need to be installed from the dependencies list.

If the image build process has started you should see some output like this:

```
    imagestream.image.openshift.io "your-application-name" created
    buildconfig.build.openshift.io "your-application-name" created
```
To see if the build process has finished, run the following command:
```
    oc logs bc/my-mlflow-experiment

```

## Second Stage: Running an experiment

After the image is built, we can use it to run an experiment on OpenShift.

To set different model training parameters and variables such as the number of CPUs and memory size, you can add the additional parameters under the `oc_run_experiment` task in the Makefile as follows:

```
    --param LIMIT_CPU=4 \
    --param LIMIT_MEM=16G \

```

Finally, you can run the experiment:

```
    make oc_run_experiment
```

This should run the experiment on OpenShift with the specified values of parameters. You can view the logs of the pod created for this experiment on OpenShift. 

If you want to use a different mlflow tracking server, you can use the MLFLOW_TRACKING_URI parameter to specify its address as:

```
    --param MLFLOW_TRACKING_URI=http://mlflow-server-url:5000/
    
```
The MLflow runs are logged remotely to a tracking server by connecting to the URI specified. 

If you know your mlflow experiment ID, you can set it with the MLFLOW_EXPERIMENT_ID environment variable.
```
    --env MLFLOW_EXPERIMENT_ID=2
```

This will group all the different runs of the model under one experiment ID.

The MLflow Tracking Server UI lets you visualize, search and compare runs, as well as download run artifacts or metadata for analysis in other tools.


## Local build

If you would like to deploy the clustering service locally, you can build the container using [S2I](https://github.com/openshift/source-to-image)

```
❯ s2i build -c . centos/python-36-centos7 aicoe-insights-clustering
```

For convenience you can store your desired environment variables in a separate file

```
❯ cat << EOT >> env.list
FLASK_ENV=development
CEPH_KEY=...
CEPH_SECRET=...
CEPH_ENDPOINT=...
EOT
```

And then run it as a Docker container

```
❯ docker run --env-file env.list  -p 8080:8080 -it aicoe-insights-clustering
```

# Data storage

Currently we support 3 types of data storage. Clustering service selects the proper one based on environment variables.

- Ceph (use `CEPH_KEY`, `CEPH_SECRET` and `CEPH_SECRET`)
- AWS S3 (use `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`)
- Local (neither variable from above is set)


**Please note that AWS S3 access is not intended to be used in development since it may touch sensitive production data.**
**For development purposes please use the local storage as described later.**


## Fetching a local copy of data from AWS

Install and configure [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html)

```
❯ pip install awscli --user
...
❯ aws configure --profile insights
AWS Access Key ID [None]: <YOUR_ACCESS_KEY>
AWS Secret Access Key [None]: <YOUR_SECRET_ACCESS_KEY>
Default region name [None]: <leave_blank>
Default output format [None]: <optional [json|text|table]>
```

Sync data locally (replace `<YYYY-MM-DD>` with an existing date). Use `DH-DEV-INSIGHTS` bucket which contains nonsensitive data only, please.

```
❯ # List available dates
❯ aws s3 ls --profile insights \
            --endpoint-url=http://storage-016.infra.prod.upshift.eng.rdu2.redhat.com:8080/ \
            s3://DH-DEV-INSIGHTS/
PRE 2018-02-28/
PRE 2018-03-01/
PRE 2018-03-02/
PRE 2018-03-03/
...

❯ # Sync to ./data
❯ aws s3 sync --profile insights \
              --endpoint-url=http://storage-016.infra.prod.upshift.eng.rdu2.redhat.com:8080/ \
              s3://DH-DEV-INSIGHTS/<YYYY-MM-DD>/rule_data ./data
```
