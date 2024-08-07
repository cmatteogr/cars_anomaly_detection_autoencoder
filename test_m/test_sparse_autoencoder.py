"""
Author: Cesar M. Gonzalez

Test Autoencoder anomaly detection model
"""
import torch
import torch.nn as nn
import pandas as pd
import json
from models.sparce_autoencoder_rl_model import SparseKLAutoencoder


def test(model_filepath: str, properties_data_filepath) -> str:
    """
    Test Autoencoder anomaly detection model
    :param model_filepath: anomaly detection model filepath to evaluate
    :param properties_data_filepath: Test data file path
    :return: test_m report
    """
    print('Start testing sparse autoencoder anomaly detection model')
    # Transform data to tensors
    properties_df = pd.read_csv(properties_data_filepath)
    n_features = len(properties_df.columns)
    tensor_data = torch.tensor(properties_df.values, dtype=torch.float32)
    rho = 0.09092432031023191

    # Apply reconstruction
    print('evaluate model on test_m data')
    model = SparseKLAutoencoder(n_features, rho)
    model.load_state_dict(torch.load(model_filepath))
    model.eval()
    criterion = nn.MSELoss()
    encoded_data, decoded_data = model(tensor_data)
    reconstruction_error = criterion(decoded_data, tensor_data)

    print(f'Reconstruction error, best score: {reconstruction_error:.8f}')
    # Build train report
    print('Create test_m report dict')
    report_dict = {
        'reconstruction_error': float(reconstruction_error)
    }
    print('save report')
    report_filepath = r'../data/test_m/test_anomaly_detection_sparse_autoencoder_report.json'
    with open(report_filepath, 'w') as json_file:
        json.dump(report_dict, json_file)

    print('End testing sparse autoencoder anomaly detection model')
    # Return model filepath
    return report_filepath
