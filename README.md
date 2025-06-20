# Cars Price Anomaly Detection (Autoencoder) - US Market

![Should Dollar Rate be an Excuse for Car Price Increase_ NO!!!](https://github.com/cmatteogr/cars_ml_project/assets/138587358/0e804a48-f2b5-4c77-a12a-16163f1244c1)

Find the best opportunities in the Car US market with desired conditions 
Cars Price Prediction project uses the data collected from [Cars Scrapy](https://github.com/cmatteogr/cars_scrapy) and use it to predict the Cars prices based on their features.

This project has been used in the Medellín Machine Learning - Study Group (MML-SG) to understand the Machine Learning bases. Therefore, the project was built end-to-end from scratch

You will find in this repo:
* Data Exploration using Jupyter to understand the data, its quality and define relevant features and their transformations.
* Model Training Pipeline:
  - Preprocess script used to apply validations, filtering, transformations, inputations, outliers removal, and normalization in the dataset.
  - Training script using multiple autoencoder models (Vanilla autoencoder, Autoencoder with Dropout, Autoencoder with L1 regularization, KL Autoencoder) all of them with the same goal: find anomalies based on desired features.
  - Evaluation script used to evaluate the model performance: RMSE, MSE.
  - Basic deployment to used the model end-to-end with new data collected 
 
## Prerequisites
* Install Python 3.11
* Install the libraries using requirements.txt.
```bash
pip install -r requirements.txt
```
* Add the cars.csv dataset CSV file (Check [Cars Scrapy](https://github.com/cmatteogr/cars_scrapy) project) in the folder .\data\data_exploration\input\

## Usage
Execute the script temp_inference.py to start the model inference.
```bash
python temp_inference.py
```
**NOTE**: Depending on the model to train the resources/time needed change so be patient or be sure you are using appropriate CPU-GPU instance.
