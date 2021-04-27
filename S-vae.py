from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from hyperspherical_vae.distributions import VonMisesFisher, HypersphericalUniform


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




