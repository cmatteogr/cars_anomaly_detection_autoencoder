import torch
import torch.nn as nn
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from models.sparce_autoencoder_kl_model import SparseKLAutoencoder

# define model parameters
properties_data_filepath = './data/preprocess/cars_test_preprocessed.csv'
properties_o_data_filepath = './data/preprocess/cars_o_test.csv'
model_filepath = './artifacts/anomaly_detection_sparce_kl_autoencoder_model_local.pth'
n_closest_anomalies = 10
# define the instance reference vector used to find the nearest neighbors for the desired anomaly
r_instance = [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# read the data
properties_df = pd.read_csv(properties_data_filepath)
# read the data original, without transformations
properties_o_df = pd.read_csv(properties_o_data_filepath)
# transform to tensor the data
n_features = len(properties_df.columns)
tensor_data = torch.tensor(properties_df.values, dtype=torch.float32)
# NOTE: rho is a hard code value, it should be replaced by a parameter. it's a hard code to explain how it works
rho = 0.09092432031023191

# apply reconstruction
print('evaluate model on evaluation data')
# load the model and weights
model = SparseKLAutoencoder(n_features, rho)
model.load_state_dict(torch.load(model_filepath))
# set the model to evaluation mode
model.eval()
# apply reconstruction
_, decoded_data = model(tensor_data)
# get the reconstruction error for each feature
reconstruction_error = (tensor_data - decoded_data).detach().numpy()

# create a Nearest Neighbors instance (finding nearest neighbors)
nn = NearestNeighbors(n_neighbors=n_closest_anomalies)
# fit the model using the reconstruction error, to find the nearest neighbors with the desired reconstruction error per feature
nn.fit(reconstruction_error)
# find the nearest neighbors
distances, indices = nn.kneighbors([r_instance])

# find the anomaly in the original data
properties_anomalies_df = properties_o_df.iloc[indices[0]]