# Hyperparameter Optimization with Ray Tune

![Modzy Logo](./imgs/modzy-banner.png)

<div align="center">

![Ray Tune & Mlflow Logo](./imgs/raytune_logo.png)


![GitHub contributors](https://img.shields.io/github/contributors/modzy/ray-tune-hyperparameter-search?logo=GitHub&style=flat)
![GitHub last commit](https://img.shields.io/github/last-commit/modzy/ray-tune-hyperparameter-search?logo=GitHub&style=flat)
![GitHub issues](https://img.shields.io/github/issues-raw/modzy/ray-tune-hyperparameter-search?logo=github&style=flat)
![GitHub](https://img.shields.io/github/license/modzy/ray-tune-hyperparameter-search?logo=apache&style=flat)

<h3 align="center">
  <a href="https://www.youtube.com/watch?v=YBJd8BQWK8Q&lc=UgzgXmoM2ApKhqBdH4B4AaABAg">Full Tech Talk on Youtube</a>
</div>


## Getting Started
This repository provides an example implementation of a Ray Tune hyperparameter search with a PyTorch training pipeline and MLflow logging. Below is an overview of the repository's contents:
* `requirements.txt`: Python packages you can pip install that are required to run the code in this repository
* `hyperparam_search.py`: Python script that loads the CIFAR dataset, defines the training pipeline, and kicks off a hyperparameter search with Ray Tune

## Environment Setup

This section provides instructions for setting up your environment and installing dependencies you will need to execute the code in this repository.

Start by cloning this project into your directory and changing the directory:

```bash
git clone https://github.com/modzy/ray-tune-hyperparameter-search.git
cd ray-tune-hyperparameter-search
```

Next, in your Python environment (**must be v3.7 or greater**), create and activate a virtual environment with your preferred virtual environment tool (conda, pyenv, venv, etc.) These instructions will leverage Python's native [venv](https://docs.python.org/3/tutorial/venv.html) module.

```bash
python -m venv venv
```

Activate environment.

*For Linux or MacOS*:

```bash
source venv/bin/activate
```

*For Windows*:

```powershell
.\venv\Scripts\activate
```

Finally, use pip to install the python packages required to run the API:

```bash
pip install -r requirements.txt
```

You are all set! Continue following along to test out the hyperparameter search yourself.

## Run Hyperparameter Search

If you'd like modify the code to add or remove parameters to tune, but to run the code as-is, simply execute the basic parameter search:

```bash
python hyperparam_search.py cifar basic_search
```
