# Digits Recognition

This is a simple digit classifier project made for a course in software enginneering for AI-enabled systems.

## Topics

### ML canvas

- [ML canvas](canvas.md)

### Cookiecutter
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This repository makes use of the CCDS project template for its directory organization.

--------

### DVC

The whole machine learning pipeline, from pre-processing to evaluation, has been turned into a DAG (Directed Acyclic Graph) the nodes of which are DVC stages that [track files](dvc.yaml) and their [modifications](dvc.lock).

Arguments of the scripts executed during the pipeline are grouped [here](params.yaml).

--------

### DagsHub

This repository is mirrored on [DagsHub](https://dagshub.com/GianmarcoTurchiano/Digits-Recognition), onto which [dataset](data/) and [model](models/) files are tracked (otherwise ignored on [GitHub](https://github.com/GianmarcoTurchiano/Digits-Recognition/)), along with experiment logs.

--------

### MLflow

Train and evaluation procedures log metrics and parameters to the [MLflow services provided by DagsHub](https://dagshub.com/GianmarcoTurchiano/Digits-Recognition.mlflow/). Runs are divided into experiments, which signify the use of different architectures.

Additionally, the trained models have been tracked as artifacts. After evaluation, when results were particularly promising, models were manually added to the [model registry](https://dagshub.com/GianmarcoTurchiano/Digits-Recognition.mlflow/).

--------

### Quality Assurance

Pylint and flake8 were used for the static analysis of the code.

Great Expectations was used to validate the size of the pictures in the dataset (see the [check stage of the pipeline](digits_recognition/experimentation/dataset/check.py)).

Pytest was employed for the following:

- [Functional tests](digits_recognition/experimentation/modeling/tests/functional_tests)

    - Tests that verify the functional correctness of the training and evaluation procedures before their are actually executed.
- [Behavioural test](digits_recognition/experimentation/modeling/tests/behavioral_tests/)
    - Multiple invariance tests verifies that there is low invariance between the predictions on original and modified pictures;
    - A minimum functionality test verifies that the model performs well on digits written using well known fonts.

--------

### APIs

FastAPI was used to implement [HTTP endpoints](digits_recognition/api/endpoints.py) that preprocess a picture of any size (even in colour) so as to classify each of the digits which may be represented within its pixels.

There are three POST endpoints. The first two endpoints, `/predictions` and `/probabilities`, both return a map where keys are x, y coordinates within the input pictures correspond respectively to a label or to a probability distribution. The third endpoint, `/annotations`, returns the input pictures with annotated predicted labels.

![image](annotations.png)

The APIs load the classifier directly from MLflow's [model registry](https://dagshub.com/GianmarcoTurchiano/Digits-Recognition.mlflow/#/models/GiaNet).

[API endpoints were also tested using Pytest](digits_recognition/api/tests/test_endpoints.py).

--------

### Dataset & Model cards

- [Dataset card](data/readme.md)
- [Model card](models/readme.md)

--------

### Docker

The APIs have been [dockerized](dockerfile) and then [composed with Prometheus and Grafana containers](docker-compose.yml).

--------

### Github Actions

<img src="https://github.com/GianmarcoTurchiano/Digits-Recognition/workflows/QA/badge.svg" />

A [job](.github/workflows/qa.yaml) which runs upon the opening of a pull request has been defined. This executes the linting macro (`make lint`), with flake8 and pylint.

--------

### Monitoring

Codecarbon was used to [log carbon emission during train and test runs](emissions.csv).

![image](training_energy.png)
![image](inference_energy.png)

Locust can be used to [stress test the server which hosts the APIs](locustfile.py).

Prometheus was setup to [log standard FastAPI metrics](digits_recognition/api/monitoring.py).

A Grafana dashboard was created to display some of those metrics.

![image](grafana_dashboard.png)

The APIs were deployed and then their uptime was tracked with Better Uptime. 

![image](uptime.png)

--------
