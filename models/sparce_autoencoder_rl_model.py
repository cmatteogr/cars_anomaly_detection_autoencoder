import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        self.num_features = num_features

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

    def kl_divergence(self, rho_hat):
        rho = torch.tensor([self.rho] * len(rho_hat)).to(rho_hat.device)
        return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

    def loss_function(self, x, decoded, encoded, beta):
        mse_loss = nn.MSELoss()(decoded, x)
        kl_loss = self.kl_divergence(torch.mean(encoded, dim=0))
        return mse_loss + beta * kl_loss