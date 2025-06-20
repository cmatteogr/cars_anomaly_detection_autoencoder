"""
Author: Cesar M. Gonzalez
Autoencoder anomaly detection model
"""

import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        self.num_features = num_features

        # this is Vanilla autoencoder
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.num_features, 84),
            nn.LeakyReLU(),
            nn.Linear(84, 58),
            nn.LeakyReLU(),
            nn.Linear(58, 40),
            nn.LeakyReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(40, 58),
            nn.LeakyReLU(),
            nn.Linear(58, 84),
            nn.LeakyReLU(),
            nn.Linear(84, self.num_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Autoencoder
        x_latent = self.encoder(x)
        reconstructed_x = self.decoder(x_latent)
        return reconstructed_x
