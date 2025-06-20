"""
Author: Cesar M. Gonzalez
Autoencoder anomaly detection model
"""
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        self.num_features = num_features

        # NOTE: encoder and decoder are the same their layer and neurons are hard code values, but they could be fine-tuned
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
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def loss_function(self, x, decoded, encoded, alpha):
        # use MSE and L1 regularization as loss function
        # alpha is the weight to apply to L1 regularization
        mse_loss = nn.MSELoss()(decoded, x)
        l1_loss = alpha * torch.norm(encoded, p=1)
        return mse_loss + l1_loss
