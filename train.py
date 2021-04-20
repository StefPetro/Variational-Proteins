import torch
import numpy as np
from misc import data, c
from torch import optim
from scipy.stats import spearmanr
from vae2 import VAE # import the last version

def get_cor_ensamble(batch, mutants_values, model, ensambles = 512, rand = True):
    model.eval()

    mt_elbos, wt_elbos = 0, 0

    for i in range(ensambles):
        if i and (i % 2 == 0):
            print(f"\tReached {i}/rand={rand}", " "*32, end="\r")

        elbos     = model.logp(batch, rand=rand).detach().cpu()
        wt_elbos += elbos[0]
        mt_elbos += elbos[1:]

    print()

    diffs  = (mt_elbos / ensambles) - (wt_elbos / ensambles)
    cor, _ = spearmanr(mutants_values, diffs)
    
    return cor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader, df, mutants_tensor, mutants_df, neff = data(batch_size = 64, device=device)

wildtype   = dataloader.dataset[0] # one-hot-encoded wildtype 
eval_batch = torch.cat([wildtype.unsqueeze(0), mutants_tensor.to(device)])

args = {
    'alphabet_len': dataloader.dataset[0].shape[0],
    'seq_len':      dataloader.dataset[0].shape[1],
    'neff':         neff
}

vae   = VAE(**args).to(device)
opt   = optim.Adam(vae.parameters())

stats = {
    'rl': [],  # rl  = Reconstruction loss
    'klz': [], # kl  = Kullback-Leibler divergence loss
    'klp': [],  # KL divergence loss for parameters
    'cor': []  # cor = Spearman correlation to experimentally measured 
    }          # protein fitness according to eq.1 from paper


for epoch in range(30):
    # Unsupervised training on the MSA sequences.
    # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    vae.train()
    
    epoch_losses = { 'rl': [], 'klp': [], 'klz': [] }
    for batch in dataloader:
        # https://discuss.pytorch.org/t/what-step-backward-and-zero-grad-do/33301/2
        opt.zero_grad()
        x_hat, mu, logvar  = vae(batch)
        loss, rl, klz, klp = vae.loss(x_hat, batch, mu, logvar)
        loss.mean().backward() # Stefan: Do we need 'retain_graph'? - ask yevgen 
        opt.step()
        epoch_losses['rl'].append(rl.mean().item())
        epoch_losses['klp'].append(klp.mean().item())
        epoch_losses['klz'].append(klz.item())

    # Evaluation on mutants
    vae.eval()
    # x_hat_eval, mu, logvar = vae(eval_batch, rep=False)
    # elbos, _, _, _ = vae.loss(x_hat_eval, eval_batch, mu, logvar)
    # elbos = vae.logp(eval_batch)
    # diffs       = elbos[1:] - elbos[0] # log-ratio (first equation in the paper)
    # cor, _      = spearmanr(mutants_df.value, diffs.detach().to('cpu'))
    cor = get_cor_ensamble(eval_batch, mutants_df.value, vae, ensambles=16, rand=True)
    
    # Populate statistics 
    stats['rl'].append(np.mean(epoch_losses['rl']))
    stats['klz'].append(np.mean(epoch_losses['klz']))
    stats['klp'].append(np.mean(epoch_losses['klp']))
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
}, "models/full_paper_2.model.pth")


