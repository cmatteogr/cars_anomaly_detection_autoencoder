"""
Author: Cesar M. Gonzalez

Autoencoder anomaly detection model
"""

import torch.nn as nn
import torch.nn.functional as functional


# define the autoencoder model
class SparseAutoencoder(nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()

        # encoder
        self.enc1 = nn.Linear(in_features=784, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=16)

        # decoder
        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=784)

    def forward(self, x):
        # encoding
        x = functional.relu(self.enc1(x))
        x = functional.relu(self.enc2(x))
        x = functional.relu(self.enc3(x))
        x = functional.relu(self.enc4(x))
        x = functional.relu(self.enc5(x))

        # decoding
        x = functional.relu(self.dec1(x))
        x = functional.relu(self.dec2(x))
        x = functional.relu(self.dec3(x))
        x = functional.relu(self.dec4(x))
        x = functional.relu(self.dec5(x))
        return x


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False
