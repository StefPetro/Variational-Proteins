# This VAE is as vanilla as it can be.
from torch import nn
import torch
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()

        self.hidden_size   = 64
        self.latent_size   = 2

        # Inputs that doesn't change
        self.alphabet_size = kwargs['alphabet_size']
        self.seq_len       = kwargs['seq_len']
        self.input_size    = self.alphabet_size * self.seq_len
            
        # Original paper: Encoder uses linear layers with 1500-1500-30 structure and ReLU transfer functions
        # Too computational heavy for deepnote. Use 64-64-32 until we find solution
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 64),  
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Latent space `mu` and `var`
        self.fc21 = nn.Linear(32, self.latent_size)
        self.fc22 = nn.Linear(32, self.latent_size)

        # Original paper: Decoder uses linear layers with size 100, ReLU, linear layer with size 2000 and sigmoid
        # and then output.
        # Too computational heavy, so we use 32-64 instead until we find solution
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid(),
            nn.Linear(64, self.input_size),
        )

    def forward(self, x, rep=True):
        x = x.view(-1, self.input_size)                    # flatten
        x = self.encoder(x)                                # encode
        mu, logvar = self.fc21(x), self.fc22(x)            # branch mu, var

        if rep:                                            # reparameterize
            x = mu + torch.randn_like(mu) * (0.5*logvar).exp() 
        else:                                              # or don't 
            x = mu                                         

        x = self.decoder(x)                                # decode
        x = x.view(-1, self.alphabet_size, self.seq_len)   # squeeze back
        x = x.log_softmax(dim=1)                           # softmax
        return x, mu, logvar
    
    def loss(self, x_hat, true_x, mu, logvar, beta=0.5):
        RL = -(x_hat*true_x).sum(-1).sum(-1)                    # reconst. loss
        KL = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(-1) # KL loss
        return RL + beta*KL, RL, KL


# RL - distance between true target and network result
# Cateogorical distribution - 250 positions - each position 20 different amino acids

# sequence weighting, two ways of doing it: 
# 1. Weighted sampling where samples are used corresponding their weight. 
# 2. Backwards loss is scaled according to weighting.
# Yevgen says: Maybe look what is best
