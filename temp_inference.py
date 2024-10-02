import torch
import torch.nn as nn
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from models.sparce_autoencoder_kl_model import SparseKLAutoencoder

properties_data_filepath = './data/preprocess/cars_test_preprocessed.csv'
properties_o_data_filepath = './data/preprocess/cars_o_test.csv'
model_filepath = './artifacts/anomaly_detection_sparce_kl_autoencoder_model_local.pth'
n_closest_anomalies = 10
r_instance = [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

properties_df = pd.read_csv(properties_data_filepath)
properties_o_df = pd.read_csv(properties_o_data_filepath)
n_features = len(properties_df.columns)
tensor_data = torch.tensor(properties_df.values, dtype=torch.float32)
rho = 0.09092432031023191

# Apply reconstruction
print('evaluate model on test_m data')
model = SparseKLAutoencoder(n_features, rho)
model.load_state_dict(torch.load(model_filepath))
model.eval()
_, decoded_data = model(tensor_data)
reconstruction_error = (tensor_data - decoded_data).detach().numpy()

# Create a Nearest Neighbors instance (finding nearest neighbors)
nn = NearestNeighbors(n_neighbors=n_closest_anomalies)
# Fit the model on the dataset
nn.fit(reconstruction_error)
# Find the nearest neighbors
distances, indices = nn.kneighbors([r_instance])

properties_anomalies_df = properties_o_df.iloc[indices[0]]
pass