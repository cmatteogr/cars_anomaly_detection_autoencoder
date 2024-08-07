import torch
import torch.nn as nn


class SparseKLAutoencoder(nn.Module):
    def __init__(self, num_features, rho):
        super(SparseKLAutoencoder, self).__init__()
        self.num_features = num_features
        self.rho = rho

        # NOTE: Sigmoid activation functions are used instead of ReLu because is convenient for the KL Divergence
        # The neural outputs are forced to return values in the range [0,1] then it simplifies compare the
        # sparsity probabilities (where the use of all the neurons are forced to find useful data patterns, not a copy)
        # * The sparsity can be understood as a distribution *
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.num_features, 84),
            nn.Sigmoid(),
            nn.Linear(84, 58),
            nn.Sigmoid(),
            nn.Linear(58, 40),
            nn.Sigmoid()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(40, 58),
            nn.Sigmoid(),
            nn.Linear(58, 84),
            nn.Sigmoid(),
            nn.Linear(84, self.num_features),
            nn.Sigmoid()
        )

    # Sparse
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def kl_divergence(self, rho_hat):
        # NOTE: Using KL divergence forces the neurons to don't activate all the time, then the neurons specialize in
        # one "representative compressed feature" of the dataset like if it was an axis in that compressed space.
        # Then all the neurons working together are able to reconstruct the inputs like if it was a distribution.
        rho = torch.tensor([self.rho] * len(rho_hat)).to(rho_hat.device)
        return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

    def loss_function(self, x, decoded, encoded, beta):
        mse_loss = nn.MSELoss()(decoded, x)
        # NOTE: The KL divergence is applied between the latent space and the input, this forces the latent space
        # neurons to specialize in a "representative compressed feature", Apply KL divergence in the output doesn't
        # have this effect. "During training, the network adjusts the weights of the neurons to minimize this
        # KL divergence, encouraging neurons to deactivate when not necessary and activate only for specific,
        # relevant features of the inputs."
        kl_loss = self.kl_divergence(torch.mean(encoded, dim=0))
        return mse_loss + beta * kl_loss