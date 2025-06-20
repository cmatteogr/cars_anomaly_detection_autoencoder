"""
Author: Cesar M. Gonzalez
Test Autoencoder anomaly detection model
"""
from models.autoencoder_model import Autoencoder
import torch
import torch.nn as nn
import pandas as pd
import json


def test(model_filepath: str, properties_data_filepath) -> str:
    """
    Test Autoencoder anomaly detection model
    :param model_filepath: anomaly detection model filepath to evaluate
    :param properties_data_filepath: Test data file path
    :return: evaluation report
    """
    print('Start testing autoencoder anomaly detection model')
    # Transform data to tensors
    properties_df = pd.read_csv(properties_data_filepath)
    n_features = len(properties_df.columns)
    tensor_data = torch.tensor(properties_df.values, dtype=torch.float32)

    # Apply reconstruction
    print('evaluate model on evaluation data')
    model = Autoencoder(n_features)
    model.load_state_dict(torch.load(model_filepath))
    # Set the model to evaluation mode
    model.eval()
    criterion = nn.MSELoss()
    # Calculate the reconstruction error
    reconstruction_data = model(tensor_data)
    reconstruction_error = criterion(reconstruction_data, tensor_data)

    print(f'Reconstruction error, best score: {reconstruction_error:.8f}')
    # Build train report
    print('Create evaluation report dict')
    report_dict = {
        'reconstruction_error': float(reconstruction_error)
    }
    print('save report')
    report_filepath = r'../data/evaluation/test_anomaly_detection_report.json'
    with open(report_filepath, 'w') as json_file:
        json.dump(report_dict, json_file)

    print('End testing autoencoder anomaly detection model')
    # Return model filepath
    return report_filepath
