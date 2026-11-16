# ML Zoomcamp Midterm Project - Stellar Objects Classification

## Task description

I wanted to work with the astronomy-related dataset, and I had a choise - to work with the exoplanet-related dataset, or with the stellar object classification dataset. In the end, I choose the stellar object classification for a simple reason - the exoplanet field is relatively new, and every year there's a new metrics to work with, which means that my model will be needing the rework constantly. On the other hand, the methods for the stellar object classification are pretty much unchanged for the last 20 to 30 years, which means that my model will be relevant for years.

## Dataset

This project uses the [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) from Kaggle.

The goal of my model is to predict whether the stellar object is a Star, a Galaxy, or a Quasar, based on 8 parameters:
1. alpha = Right Ascension angle (at J2000 epoch)
2. delta = Declination angle (at J2000 epoch)
3. u = Ultraviolet filter in the photometric system
4. g = Green filter in the photometric system
5. r = Red filter in the photometric system
6. i = Near Infrared filter in the photometric system
7. z = Infrared filter in the photometric system
8. redshift = redshift value based on the increase in wavelength

## Model Training

Multiple models were tried before picking the best one - such as Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting using XGBoost.
While all models showed great results in predicting the class of an object, in the end, the Gradient Boosting was the best choise, even if by a small margin of 0.2% - you can find the notebook.ipynb file in the project directory, which includes the following steps:
 * The data preparation and data cleaning; 
 * EDA, Feature Importance, Correlation Matrix and Mutual Info;
 * Data Scaling, Model Selection and Parameter Tuning for LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, and XGBoost;
 * Final thoughts and comparison of the precision of each model

## Project Structure

```
midterm_project/
├── notebook.ipynb            # jupyter notebook, which includes all the steps from the Model Training readme section
├── star_classification.csv   # dataset from kaggle
├── train.py                  # python script that trains XGBoost model and saves it to .bin file
├── predict.py                # Flask skript that loads the model from .bin file and predicts the stellar class type
├── predict-test.py           # test script to send a sample request
├── xgb_model_depth_6.bin     # saved model, generated after running the predict.py script
├── Pipfile                   # pipenv dependencies
├── Pipfile.lock              # pipenv dependency versions
└── Dockerfile                # list of instructions, from which a Docker image is built
```

## Prerequisites

Before running anything, make sure you have the following installed:

- **Python 3.11** — download from https://www.python.org/downloads/
- **pip** — comes bundled with Python 3.11, no separate install needed
- **Docker** — download from https://www.docker.com/products/docker-desktop/ (only needed for the Docker setup)

## To run things locally on Windows machines:

### 1. Install dependencies

pip install pipenv

pipenv install

### 2. Train the model from the pipenv, which will create the model .bin file in the project directory

pipenv run python train.py

### 3. Run the Flask web servise from the predict.py file

pipenv run python predict.py

### 4. In a separate terminal, run a predict-test.py script

python predict-test.py

## To run everything in a Docker container on Windows machines:

### 1. Install pipenv

pip install pipenv

pipenv install

### 2. Train the model, if you don't have the .bin file in the project directory

pipenv run python train.py

### 3. Download the Docker 

Follow the instruction by Andrew Lock: https://andrewlock.net/installing-docker-desktop-for-windows/.

### 4. Build the Docker image from the Dockerfile in the project directory

docker build -t stellar-classification .

### 5. Run the Docker container

docker run -it --rm -p 9696:9696 stellar-classification

### 6. In a separate terminal, run a predict-test.py script

python predict-test.py

#### Note: you might need to install the request library by running ```pip install requests```, or ```pipenv install requests```, if you are using the Python virtual environment

## After running the predict-test.py script(either locally or using the Docker container), send a POST request to `http://localhost:9696/predict` with a JSON body containing the following fields:

```json
{
    "alpha": 135.689,
    "delta": 32.494,
    "u": 19.2,
    "g": 18.1,
    "r": 17.9,
    "i": 17.7,
    "z": 17.5,
    "redshift": 0.5
}
```

The order of these fields doesn't matter

The response will look something like this:

```json
{
    "predicted_class": "QSO",
    "probabilities": {
        "GALAXY": 0.064,
        "STAR": 0.0005,
        "QSO": 0.934
    }
}
```

`predicted_class` can be one of three types: `GALAXY`, `STAR`, or `QSO`.
