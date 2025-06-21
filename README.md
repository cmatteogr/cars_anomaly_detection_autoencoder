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

## External Resoruces
This project was built by the Medellín Machine Learning - Study Group (MML-SG) community. In the following [link](https://drive.google.com/drive/u/0/folders/1nPMtg6caIef5o9S_J8WyNEvyEt5sO1VH) you can find the meetings records about it:
* [2. Exploración de Modelos de ML y Exploración de Datos (2024-02-28 19:14 GMT-5)](https://drive.google.com/file/d/1mqpccGVjhOQTDV5c80RKk1ECNnK6DCqn/view?usp=drive_link)
* [6. Implementación de la Detección de Anomalías (2024-04-17 19:09 GMT-5)](https://drive.google.com/file/d/1NU6CLKnL_O4xxduqQlrtPCgiFCZQKiI4/view?usp=drive_link)
* [7. Evaluación del Modelo y Resultados de la Detección de Anomalías (2024-04-24 19:09 GMT-5)](https://drive.google.com/file/d/1IFQ1AFlBal3UAFbdfB474GRovBOQUXaw/view?usp=drive_link)
* [11. Introducción al Uso de Autoencoders en Detección de Anomalías (2024-05-22 19:34 GMT-5)](https://drive.google.com/file/d/1oeU1jpZyS1LKMy5NJ1kWx7D-3LxFQPV8/view?usp=drive_link)
* [12. Implementación de Autoencoders para Detección de Anomalías en Datos de Carros (2024-05-29 19:11 GMT-5)](https://drive.google.com/file/d/1ToR7FLbmEpnniLP40j-qmrRz8IR5OYw6/view?usp=drive_link)
* [13. Evaluación del Desempeño de Autoencoders en la Detección de Anomalías (2024-06-05 19:10 GMT-5)](https://drive.google.com/file/d/10gnKyJxK4q7M5Bq965K0M5P9ZsydtUHi/view?usp=drive_link)
* [14. Optimización de Autoencoders para Mejorar la Detección de Anomalías (2024-06-12 19:10 GMT-5)](https://drive.google.com/file/d/12kgw_zjYcCLM52kB3wU5LRIDnidPVeU7/view?usp=drive_link)
* [15. Integración de Autoencoders en el Pipeline de Predicción de Precios de Carros (2024-06-18 19:12 GMT-5)](https://drive.google.com/file/d/1IaGO2UB1eqnhjdQbupemnm7UDYJMrgEK/view?usp=drive_link)
* [18. Revisión Final y Lecciones Aprendidas sobre Autoencoders para Detección de Anomalías (2024-08-07 19:09 GMT-5)](https://drive.google.com/file/d/1y7pZ43Ss5RF_Znh2LVQ3o-f2T7ni6jie/view?usp=drive_link)
