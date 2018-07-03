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
