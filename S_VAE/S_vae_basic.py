import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

import numpy as np
import torch.optim as optim
import torch.utils.data
from collections import defaultdict

from hyperspherical_vae.distributions import VonMisesFisher, HypersphericalUniform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Basic_S_VAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Basic_S_VAE, self).__init__()

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
        # compute mean and concentration of the von Mises-Fisher
        self.fc21 = nn.Linear(self.hidden_size, self.latent_size) 
        self.fc22 = nn.Linear(self.hidden_size, 1)

        self.fc3 = nn.Linear(self.latent_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.input_size)


    def encode(self, x):
        x = x.view(-1, self.input_size)          # flatten input
        x = self.fc1(x)                          # input to hidden size
        x = F.relu(x)                            # ReLU activation

        mu, logvar = self.fc21(x), self.fc22(x)  # branch mu, var

        # From S-VAE example - https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py
        mu = mu / mu.norm(dim=-1, keepdim=True)
        # the `+ 1` prevent collapsing behaviors
        # Softplus to avoid zeroes in the VMG and HYU distributions
        logvar = F.softplus(logvar) + 1

        return mu, logvar


    def reparameterize(self, mu, logvar):
        q_z = VonMisesFisher(mu, logvar)
        p_z = HypersphericalUniform(self.latent_size - 1)
        return q_z, p_z


    def decode(self, z):
        h = self.fc3(z)       # latent size to hidden size
        h = F.relu(h)         # ReLU activation
        out = self.fc_out(h)  # hidden size to input size
        return out


    def forward(self, x):
        mu, logvar = self.encode(x)                            # encode
        mu, logvar = mu.detach().cpu(), logvar.detach().cpu()  # detach and set to cpu because VMF and HYU  

        q_z, p_z = self.reparameterize(mu, logvar)             # reparameterize
        z = q_z.rsample().to(device)                           # sample from VMF, set to GPU again

        out = self.decode(z)                                  # decode
        out = out.view(-1, self.alphabet_len, self.seq_len)    # squeeze back
        out = out.log_softmax(dim=1)                           # softmax

        return out, mu, logvar, q_z, p_z


    def loss(self, recon_x, x, mu, logvar, q_z, p_z):
        # RL, Reconstruction loss
        RL = (-recon_x*x).sum(-2).sum(-1)

        # KL, Kullback-Leibler divergence loss (for `z`):
        KLZ   = kl_divergence(q_z, p_z).sum(-1)

        loss = (RL + self.beta * KLZ)
        
        return loss, RL.mean(), KLZ.mean()