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
        * GIT_URI=https://github.com/RedHatInsights/aicoe-insights-clustering.git

--> Creating resources ...
    buildconfig "systems-clustering" created
    imagestream "systems-clustering" created
--> Success
    Use 'oc start-build systems-clustering' to start a build.
    Run 'oc status' to view your app.
```

Start a build

```
❯ oc start-build systems-clustering
build "systems-clustering-1" started
```

And finally you can run a job that does the clustering

```
❯ oc new-app --template systems-clustering-pod --param CEPH_KEY=I..................8 --param CEPH_SECRET=g......................................A --param CEPH_ENDPOINT=http://storage-.......................................com:8080/
--> Deploying template "mhild-test/systems-clustering-pod" to project mhild-test

     * With parameters:
        * JOB_NAME=systems-clustering-job-spah # generated
        * LABEL_APP_NAME=systems-clustering
        * CLUSTER_IMAGE=systems-clustering
        * CEPH_KEY=I..................8
        * CEPH_SECRET=g......................................A
        * CEPH_ENDPOINT=http://storage-.......................................com:8080/

--> Creating resources ...
    pod "systems-clustering-job-spah" created
--> Success
    Run 'oc status' to view your app.
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
