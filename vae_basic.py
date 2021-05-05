from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


class Basic_VAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Basic_VAE, self).__init__()

        # Layer sizes
        self.hidden_size = kwargs.get('hidden_size', 64)
        self.latent_size = kwargs.get('latent_size', 2)
        print(f'Size of latent space: {self.latent_size}')

        # Inputs that don't change
        self.alphabet_len  = kwargs['alphabet_len']
        self.seq_len       = kwargs['seq_len']
        self.input_size    = self.alphabet_len * self.seq_len

        # .get() sets a value if the key doesn't exist
        self.beta = kwargs.get('beta', 0.5)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)

        # Latent space `mu` and `var`
        self.fc21 = nn.Linear(self.hidden_size, self.latent_size) 
        self.fc22 = nn.Linear(self.hidden_size, self.latent_size)

        self.fc3 = nn.Linear(self.latent_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.input_size)

    
    def encode(self, x):
        x = x.view(-1, self.input_size)          # flatten input
        x = self.fc1(x)                          # input to hidden size
        x = F.relu(x)                            # ReLU activation

        mu, logvar = self.fc21(x), self.fc22(x)  # branch mu, var
        return mu, logvar
    

    def reparameterize(self, mu, logvar):
        z = mu + torch.randn_like(mu) * (0.5*logvar).exp()
        return z


    def decode(self, z):
        h = self.fc3(z)       # latent size to hidden size
        h = F.relu(h)         # ReLU activation
        out = self.fc_out(h)  # hidden size to input size
        return out


    def forward(self, x):
        mu, logvar = self.encode(x)                          # encode      
        z = self.reparameterize(mu, logvar)                  # reparameterize
        out = self.decode(z)                                 # decode

        out = out.view(-1, self.alphabet_len, self.seq_len)  # squeeze back
        out = out.log_softmax(dim=1)                         # softmax

        return out, mu, logvar


    def loss(self, recon_x, x, mu, logvar):
        # RL, Reconstruction loss
        RL = (-recon_x*x).sum(-2).sum(-1)

        # KL, Kullback-Leibler divergence loss (for `z`):
        KLZ = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)

        loss = (RL + self.beta * KLZ)
        return loss, RL.mean(), KLZ.mean()
