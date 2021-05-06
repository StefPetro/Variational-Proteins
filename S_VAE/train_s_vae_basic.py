import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from misc import data, c
from torch import optim
from scipy.stats import spearmanr
from S_vae_basic import Basic_S_VAE # import the last version

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader, df, mutants_tensor, mutants_df = data(batch_size = 64, device=device, weighted_sampling=False)

wildtype   = dataloader.dataset[0] # one-hot-encoded wildtype 
eval_batch = torch.cat([wildtype.unsqueeze(0), mutants_tensor.to(device)])

args = {
    'alphabet_len': dataloader.dataset[0].shape[0],
    'seq_len':      dataloader.dataset[0].shape[1],
    'latent_size':  2,
    'hidden_size':  64,
}

vae   = Basic_S_VAE(**args).to(device) # Hyperspherical VAE
opt   = optim.Adam(vae.parameters())

stats = {
    'rl': [],  # rl  = Reconstruction loss
    'klz': [], # kl  = Kullback-Leibler divergence loss
    'cor': []  # cor = Spearman correlation to experimentally measured 
    }          # protein fitness according to eq.1 from paper


for epoch in range(200):
    # Unsupervised training on the MSA sequences.
    # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    vae.train()
    
    epoch_losses = { 'rl': [], 'klp': [], 'klz': [] }
    for batch in dataloader:
        # https://discuss.pytorch.org/t/what-step-backward-and-zero-grad-do/33301/2
        opt.zero_grad()
        x_hat, mu, logvar, q_z, p_z = vae(batch)
        loss, rl, klz = vae.loss(x_hat, batch, mu, logvar, q_z, p_z)
        loss.mean().backward()
        opt.step()
        epoch_losses['rl'].append(rl.mean().item())
        epoch_losses['klz'].append(klz.item())

    # Evaluation on mutants
    vae.eval()
    x_hat_eval, mu, logvar, q_z, p_z = vae(eval_batch)
    elbos, _, _ = vae.loss(x_hat_eval, eval_batch, mu, logvar, q_z, p_z)
    diffs       = elbos[1:] - elbos[0] # log-ratio (first equation in the paper)
    cor, _      = spearmanr(mutants_df.value, diffs.detach().cpu())
    
    # Populate statistics 
    stats['rl'].append(np.mean(epoch_losses['rl']))
    stats['klz'].append(np.mean(epoch_losses['klz']))
    stats['cor'].append(np.abs(cor))

    to_print = [
        f"{c.HEADER}EPOCH %03d"          % epoch,
        f"{c.OKBLUE}RL=%4.4f"            % stats['rl'][-1], # reconstrution loss
        f"{c.OKGREEN}KLZ=%4.4f"          % stats['klz'][-1], # KL loss
        f"{c.OKCYAN}|rho|=%4.4f{c.ENDC}" % stats['cor'][-1] # correlation (we want to max this value)
    ]
    print(" ".join(to_print))

torch.save({
    'state_dict': vae.state_dict(), 
    'stats':      stats,
    'args':       args,
}, "models/Basic_SVAE_ep200_hs64_ls2.model.pth") # ep = epochs, hs = hidden size, e = ensamble, ls = latent size


