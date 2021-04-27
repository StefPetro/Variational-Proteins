from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_new_weight(module, input):
    m = module

    if ('bias' in module._parameters.keys()) and (module.bias is not None):
        has_bias = True
    else:
        has_bias = False

    # Weight
    # std = module.weight_logvar.mul(1/2).exp()
    # eps = torch.randn_like(std)
    # setattr(module, "weight", module.weight_mean + eps*std)
    setattr(module, "weight", Normal(m.weight_mean, m.weight_logvar.mul(1/2).exp()).rsample())

    # Bias
    # std = torch.exp(0.5 * module.bias_logvar)
    # eps = torch.radn_like(std)
    # setattr(module, "bias", module.bias_mean + eps*std)
    if has_bias:
        setattr(module, "bias", Normal(m.bias_mean, m.bias_logvar.mul(1/2).exp()).rsample())
    else:
        setattr(module, "bias", None)

    return None 
    
def make_variational_linear(module, name=None):
    m, has_bias = module, module.bias is not None
    setattr(module, "name", name)

    del m._parameters['weight']

    if has_bias:
        del m._parameters['bias']

    weight_mean_param   = nn.Parameter(torch.Tensor(m.out_features, m.in_features))
    weight_logvar_param = nn.Parameter(torch.Tensor(m.out_features, m.in_features)) 

    m.register_parameter('weight_mean', weight_mean_param)
    m.register_parameter('weight_logvar', weight_logvar_param)

    variance = 2 / (m.out_features + m.in_features)
    nn.init.normal_(weight_mean_param, 0.0, std = variance**(1/2))
    nn.init.constant_(weight_logvar_param, -5) # WAS -10

    if has_bias:
        bias_mean_param    = nn.Parameter(torch.Tensor(m.out_features))
        bias_logvar_param  = nn.Parameter(torch.Tensor(m.out_features))

        m.register_parameter('bias_mean', bias_mean_param)
        m.register_parameter('bias_logvar', bias_logvar_param)

        nn.init.constant_(bias_mean_param, 0.1)
        nn.init.constant_(bias_logvar_param, -5) # WAS -10

    m.register_forward_pre_hook(sample_new_weight) # https://www.kite.com/python/docs/torch.nn.Module.register_forward_pre_hook
    sample_new_weight(m, None)

    return m


class VAE(nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()

        self.variational_layers = []

        # Layer sizes
        self.hidden_size = 2000
        self.latent_size = kwargs.get('latent_size', 32)
        print(f'Size of latent space: {self.latent_size}')

        # Inputs that don't change
        self.alphabet_len  = kwargs['alphabet_len']
        self.seq_len       = kwargs['seq_len']
        self.neff          = kwargs['neff'] # P: what is this?
        self.input_size    = self.alphabet_len * self.seq_len

        # .get() sets a value if the key doesn't exist
        self.div = kwargs.get('div', 8)
        self.beta = kwargs.get('beta', 1)
        self.inner = kwargs.get('inner', 16) 
        self.h2_div = kwargs.get('h2_div', 1)
        self.bayesian = kwargs.get('bayesian', True)
        self.dropout = kwargs.get('dropout', 0.0)

        self.lamb = nn.Parameter(torch.Tensor([0.1] * self.input_size))  # lambda - regularization term
        self.W_out_b = nn.Parameter(torch.Tensor([0.1] * self.input_size))  # bias
        
        # Original paper: Encoder uses linear layers with 1500-1500-30 structure and ReLU transfer functions
        self.fc1 = nn.Linear(self.input_size, int(self.hidden_size * (3/4)))
        self.fc1h = nn.Linear(int(self.hidden_size * (3/4)), int(self.hidden_size * (3/4)))

        # Latent space `mu` and `var`
        self.fc21 = nn.Linear(int(self.hidden_size * (3/4)), self.latent_size) 
        self.fc22 = nn.Linear(int(self.hidden_size * (3/4)), self.latent_size)

        # Original paper: Decoder uses linear layers with size 100, ReLU, linear layer with size 2000 and sigmoid
        # and then output.
        self.fc3 = nn.Linear(self.latent_size, self.hidden_size // 16)
        self.fc3 = make_variational_linear(self.fc3)
        self.variational_layers.append(self.fc3)

        self.fc3h = nn.Linear(self.hidden_size // 16, self.hidden_size // self.h2_div)
        self.fc3h = make_variational_linear(self.fc3h)
        self.variational_layers.append(self.fc3h)

        # For the Group Sparsity Prior
        self.W = nn.Linear(self.inner, (self.hidden_size // self.h2_div) * self.seq_len, bias=False)  # Weights
        self.C = nn.Linear(self.alphabet_len, self.inner, bias=False)  # "compressing" layer
        self.S = nn.Linear(self.seq_len, (self.hidden_size // self.h2_div) // self.div, bias=False)  # Scaling

        if self.bayesian:
            self.W = make_variational_linear(self.W, 'W')
            self.C = make_variational_linear(self.C, 'C')
            self.S = make_variational_linear(self.S, 'S')
            self.variational_layers.append(self.W)
            self.variational_layers.append(self.C)
            self.variational_layers.append(self.S)

            lamb_W_out_b_dim = self.input_size

            self.lamb_mean      = nn.Parameter(torch.Tensor([1]   * 1))
            self.lamb_logvar    = nn.Parameter(torch.Tensor([-5]  * 1))
            self.W_out_b_mean   = nn.Parameter(torch.Tensor([0.1] * lamb_W_out_b_dim))
            self.W_out_b_logvar = nn.Parameter(torch.Tensor([-5]  * lamb_W_out_b_dim))
        else:
            self.lamb    = nn.Parameter(torch.Tensor([0.1 * self.input_size])) # NOTE: WORKS
            self.W_out_b = nn.Parameter(torch.Tensor([0.1 * self.input_size])) # NOTE: WORKS
            # 0.01 * torch.randn(self.input_size).abs()

        # INIT
        # Using article: https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
        n = 0 # count of linear layers
        for layer in self.children():
            if not str(layer).startswith('Linear'):
                continue
            
            if n == 2 or n == 3 or n => 5: # making sure the init is on latent, sigmoid activation and group sparsity
                nn.init.xavier_normal_(layer.weight)  # Xavier normalization for layers with Sigmoid activation
                print(f'{layer} init Xavier')
            else:
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu') # Kaiming normalization for layers with RELU activation
                print(f'{layer} init Kaiming')
            
            # init layers bias
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.1)  # article says init for Kaiming should be 0.0, but that may cause problems.
                                                    # See: https://github.com/loliverhennigh/Variational-autoencoder-tricks-and-tips/blob/master/README.md
            
            n += 1
        
        nn.init.constant_(self.fc22.bias, -5)


    def encode(self, x):    
        x = x.view(-1, self.input_size)                    # flatten

        # Encode
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1h(x)
        x = F.relu(x)

        mu, logvar = self.fc21(x), self.fc22(x)            # branch mu, var

        return mu, logvar

    def reparameterize(self, mu, logvar):
        z = mu + torch.randn_like(mu) * (0.5*logvar).exp() # reparameterize
        
        return z
    
    def decode(self, x):
        # Decode
        h = self.fc3(x)
        h = F.relu(h)
        h = self.fc3h(h)
        h = torch.sigmoid(h)

        if self.bayesian:
            for module in [self.W, self.C, self.S]: # We need to sample the weight manually 
                sample_new_weight(module, None)     # because we never call forward on those
                                                    # layers

            lamb = Normal(self.lamb_mean,       self.lamb_logvar.mul(1/2).exp()).rsample()
            b    = Normal(self.W_out_b_mean, self.W_out_b_logvar.mul(1/2).exp()).rsample()    
        else:
            lamb = self.lamb                    # lamb:  [ input_size                 ]
            b    = self.W_out_b                 # b:     [ input_size                 ]

        W = self.W.weight                       # W:    [ (hidden x seq_len) x inner ]
        C = self.C.weight                       # C:    [ inner   x alphabet_len     ]
        S = self.S.weight.repeat(self.div, 1)   # S:    [ hidden  x seq_len          ]
        S = torch.sigmoid(S)                    # S:    [ hidden  x seq_len          ]

        W_out = torch.mm(W, C)                  # W_out: [ (hidden x seq_len) x alphabet_len ]
        order = (self.hidden_size // self.h2_div, self.alphabet_len, self.seq_len)
        W_out = W_out.view(*order)              # W_out: [ hidden x alphabet_len x seq_len ]
        S = S.unsqueeze(-2)                     # S:     [ hidden x 1 x seq_len ]
        W_out = W_out * S                       # W_out: [ hidden x alphabet_len x seq_len ]
        W_out = W_out.view(-1, self.input_size) # W_out: [ hidden x input_size ]

        return (1 + self.lamb.exp()).log() * F.linear(h, W_out.T, self.W_out_b)


    def forward(self, x, rep=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)                                   # decode
        
        # P: what the two lines below do?
        out = out.view(-1, self.alphabet_len, self.seq_len)    # squeeze back
        out = out.log_softmax(dim=1)                           # softmax

        return out, mu, logvar
    

    def logp(self, batch, rand = False):
        ''' Returns individual log likelihoods of the batch'''

        mu, logvar = self.encode(batch)
        z          = self.reparameterize(mu, logvar) if rand else mu

        recon      = self.decode(z)
        recon      = recon.view(-1, self.alphabet_len, self.seq_len)
        recon      = recon.log_softmax(dim=1)

        # logp     = F.nll_loss(recon, batch.argmax(dim=1), reduction='none').sum(-1)
        logp       = (-recon*batch).sum(-2).sum(-1)
        kl         = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
        elbo       = logp + kl

        return elbo


    def loss(self, recon_x, x, mu, logvar):
        # RL, Reconstruction loss
        # RL = F.nll_loss(recon_x, x.argmax(dim=1), reduction='none).sum(-1)
        RL = (-recon_x*x).sum(-2).sum(-1)
        # RL2 = F.nll_loss(recon_x, x.argmax(dim=1), reduction='none').sum(-1)

        # KL, Kullback-Leibler divergence loss (for `z`):
        # KLZ = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
        KLZ_q = Normal(mu, logvar.mul(1/2).exp())
        KLZ_p = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        KLZ   = kl_divergence(KLZ_q, KLZ_p).sum(-1)

        # KLP, Kullback-Leibler divergence loss (for network parameters):
        KLP = 0
        for l in self.variational_layers:
            weight_mu, weight_std = l.weight_mean, l.weight_logvar.mul(1/2).exp()
            q_weight = Normal(weight_mu, weight_std)

            if l.name == 'S':
                p_weight = Normal(torch.zeros_like(weight_mu) - 9.305, torch.zeros_like(weight_std) + 4)
            else:
                p_weight = Normal(torch.zeros_like(weight_mu), torch.ones_like(weight_std))
            
            KLP += kl_divergence(q_weight, p_weight).sum()

            if l.bias is None: continue

            bias_mu, bias_std = l.bias_mean, l.bias_logvar.mul(1/2).exp()
            q_bias = Normal(bias_mu, bias_std)
            p_bias = Normal(torch.zeros_like(bias_mu), torch.ones_like(bias_std))
            KLP += kl_divergence(q_bias, p_bias).sum()


        # Variational parameters: lambda (lamb) and w_out_b
        lamb_mu, lamb_std = self.lamb_mean, self.lamb_logvar.mul(1/2).exp()
        q_lamb = Normal(lamb_mu, lamb_std)
        p_lamb = Normal(torch.zeros_like(lamb_mu), torch.ones_like(lamb_std))
        KLP   += kl_divergence(q_lamb, p_lamb).sum()

        W_out_b_mu, W_out_b_std = self.W_out_b_mean, self.W_out_b_logvar.mul(1/2).exp()
        q_W_out_b = Normal(W_out_b_mu, W_out_b_std)
        p_W_out_b = Normal(torch.zeros_like(W_out_b_mu), torch.ones_like(W_out_b_std))
        KLP      += kl_divergence(q_W_out_b, p_W_out_b).sum()

        KLP /= self.neff
        loss = (RL + self.beta * KLZ).mean() + KLP

        KLP = torch.tensor([0], requires_grad=False) if KLP == 0 else KLP
        
        return loss, RL.mean(), KLZ.mean(), KLP


# RL - distance between true target and network result
# Cateogorical distribution - 250 positions - each position 20 different amino acids

# sequence weighting, two ways of doing it: 
# 1. Weighted sampling where samples are used corresponding their weight. 
# 2. Backwards loss is scaled according to weighting.
# Yevgen says: Maybe look what is best
