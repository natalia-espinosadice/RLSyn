import time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os 
from pathlib import Path
import numpy as np 
import argparse 
from evaluation.evaluate_aireadi import (log_result_RL, compute_value_stats, print_stats, latent_cluster_analysis, train_on_real_test_on_real, train_on_synth_test_on_hold, train_on_hold_test_on_synth, get_histograms, get_column_wise_correlations, train_on_real_test_on_synth, train_on_synth_test_on_real, classify_real_vs_syn, plot_rl_losses, mem_risk)
from evaluation.evaluate_mimic import (get_column_wise_correlationsM, compute_dimension_wide_distribution, latent_cluster_analysisM, run_PCA, medical_concept_abundance, clinical_knowledge_violation, train_and_test_classification, log_result_RL_MIMIC, mem_risk_2only, mem_risk_MIMIC)
from RL.hyperparams import HyperParams_AIREADI, HyperParams_MIMIC
from RL.models import build_models
import random

def set_seed(seed: int = 42) -> None:
    random.seed(seed)                      
    np.random.seed(seed)          
    torch.manual_seed(seed)        
    torch.cuda.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 



def REINFORCE(df_train, real, loader, H): 
    start_time = time.time() 
    #instantiate
    G, D = build_models(H)
    opt_G = torch.optim.Adam(G.parameters(), lr=H.G_LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=H.D_LR)
    os.makedirs(f"{H.OUT_DIR}/losses", exist_ok=True)
    with open(f"{H.OUT_DIR}/losses/output.txt", "w") as f:
        f.write(f"Logging\n")
    with open(f"{H.OUT_DIR}/losses/G_loss.txt", "w") as f:
        f.write(f"Logging\n")
    #ppo training loop
    real_iter = iter(loader)
    for it in range(H.ITERS):
        #rollout policy 
        z = torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE)
        rows, _, _, _ = G.sample(z)
        #detach for grad
        rows = rows.detach()      
        with torch.no_grad():
            rewards = torch.sigmoid(D(rows)).squeeze()
        #REINFORCE UPDATE
        logp, _ = G.eval_action(z, rows)
        adv_n = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        loss_G = -(logp * adv_n).mean() 
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
        #with open(f"{H.OUT_DIR}/losses/G_loss.txt", "a") as f:
        #    f.write(f"ITERATION {it:.4f} | TOTAL G LOSS {loss_G:.4f}\n")
        #discriminator update 
        for d_it in range(H.DISC_STEPS):
            try:
                real_batch, = next(real_iter)
            except StopIteration:
                real_iter = iter(loader)
                real_batch, = next(real_iter)
            real_batch = real_batch.to(H.DEVICE)
            #fresh fake batch
            fake_batch, _, _, _ = G.sample(torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE))
            fake_batch = fake_batch.detach()
            #R1 gradient penalty on real data --> not quite WGAN!!! Could try with wasserstein distance 
            real_batch.requires_grad_(True)
            real_logits = D(real_batch)
            grad_real = torch.autograd.grad(real_logits.sum(), real_batch, create_graph=True)[0]
            gp = H.GRADIENT_PENALTY * 0.5 * grad_real.pow(2).view(real_batch.size(0), -1).sum(1).mean()
            loss_D = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_batch[:, :1])) + F.binary_cross_entropy_with_logits(D(fake_batch), torch.zeros_like(fake_batch[:, :1])) + gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
        if it % 5000 == 0:
            print(f"{it} complete")
            with open(f"{H.OUT_DIR}/losses/output.txt", "a") as f:
                f.write(f"iteration {it} | D LOSS = {loss_D.item():.4f} | G LOSS = {loss_G.item():.4f} | AVG R ={rewards.mean():.4f} |  \n")
    #generate and save  
    z = torch.randn(H.NUM_SAMPLES, H.NOISE_DIM, device=H.DEVICE)
    synthetic, _, _, _ = G.sample(z)
    cols = H.NUM_COLS + H.CAT_COLS
    if H.DEVICE == "cuda": 
        df_syn = pd.DataFrame(synthetic.cpu().detach().numpy(), columns=cols)
    else: 
        df_syn = pd.DataFrame(synthetic.detach().numpy(), columns=cols)
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic.csv")
    #rescale 
    feature_range = np.load(H.NPY_PATH, allow_pickle=True).item()
    for col in H.NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")
    elapsed_time = (time.time() - start_time) / 60 # minutes 
    return df_syn, elapsed_time

def reparam_GAN(df_train, real, loader, H): 
    start_time = time.time() 
    #instantiate
    G, D = build_models(H)
    opt_G = torch.optim.Adam(G.parameters(), lr=H.G_LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=H.D_LR)
    os.makedirs(f"{H.OUT_DIR}/losses", exist_ok=True)
    with open(f"{H.OUT_DIR}/losses/output.txt", "w") as f:
        f.write(f"Logging\n")
    with open(f"{H.OUT_DIR}/losses/G_loss.txt", "w") as f:
        f.write(f"Logging\n")
    real_iter = iter(loader)
    for it in range(H.ITERS):
        #gen - gradient flows 
        z = torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE)
        rows, _, _, _ = G.sample(z)  #NOT DEATHCED 
        fake_logits = D(rows)
        loss_G = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        #mean penalty 
        target_mean = real.mean(0, keepdim=True).to(H.DEVICE)
        num_fake = rows[:, :len(H.NUM_COLS)]
        mean_pen = (num_fake.mean(0) - target_mean[0, :len(H.NUM_COLS)]).pow(2).mean()
        loss_G += H.MEAN_PENALTY_SCALE * mean_pen 
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
        with open(f"{H.OUT_DIR}/losses/G_loss.txt", "a") as f:
            f.write(f"ITERATION {it:.4f} | MEAN PEN (*.2) {(H.MEAN_PENALTY_SCALE * mean_pen):.4f} | TOTAL G LOSS {loss_G:.4f}\n")
        #disc same 
        for d_it in range(H.DISC_STEPS):
            try:
                real_batch, = next(real_iter)
            except StopIteration:
                real_iter = iter(loader)
                real_batch, = next(real_iter)
            real_batch = real_batch.to(H.DEVICE)
            with torch.no_grad():
                fake_batch, _, _, _ = G.sample(torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE))
            fake_batch = fake_batch.detach()
            real_batch.requires_grad_(True)
            real_logits = D(real_batch)
            grad_real = torch.autograd.grad(real_logits.sum(), real_batch, create_graph=True)[0]
            gp = H.GRADIENT_PENALTY * 0.5 * grad_real.pow(2).view(real_batch.size(0), -1).sum(1).mean()
            loss_D = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_batch[:, :1])) + F.binary_cross_entropy_with_logits(D(fake_batch), torch.zeros_like(fake_batch[:, :1])) + gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
        if it % 50 == 0:
            print(f"{it} complete")
            with open(f"{H.OUT_DIR}/losses/output.txt", "a") as f:
                f.write(f"iteration {it} | D LOSS = {loss_D.item():.4f} | G LOSS = {loss_G.item():.4f} | mean_pen: {(mean_pen.item()*H.MEAN_PENALTY_SCALE):.4f} |  \n")
    #generate and save  
    z = torch.randn(H.NUM_SAMPLES, H.NOISE_DIM, device=H.DEVICE)
    synthetic, _, _, _ = G.sample(z)
    cols = H.NUM_COLS + H.CAT_COLS
    if H.DEVICE == "cuda": 
        df_syn = pd.DataFrame(synthetic.cpu().detach().numpy(), columns=cols)
    else: 
        df_syn = pd.DataFrame(synthetic.detach().numpy(), columns=cols)
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic.csv")
    #rescale 
    feature_range = np.load(H.NPY_PATH, allow_pickle=True).item()
    for col in H.NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")
    elapsed_time = (time.time() - start_time) / 60
    return df_syn, elapsed_time


def reparam_GAN_cat(df_train, real, loader, H): 
    start_time = time.time() 
    G, D = build_models(H)
    opt_G = torch.optim.Adam(G.parameters(), lr=H.G_LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=H.D_LR)
    os.makedirs(f"{H.OUT_DIR}/losses", exist_ok=True)
    with open(f"{H.OUT_DIR}/losses/output.txt", "w") as f:
        f.write(f"Logging\n")
    with open(f"{H.OUT_DIR}/losses/G_loss.txt", "w") as f:
        f.write(f"Logging\n")
    real_iter = iter(loader)
    for it in range(H.ITERS):
        #gen 
        z = torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE)
        rows = G.sample_reparam(z) #NOT DETACHED NOW 
        fake_logits = D(rows)
        loss_G = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        #mean penalty 
        target_mean = real.mean(0, keepdim=True).to(H.DEVICE)
        num_fake = rows[:, :len(H.NUM_COLS)]
        mean_pen = (num_fake.mean(0) - target_mean[0, :len(H.NUM_COLS)]).pow(2).mean()
        #loss_G += H.MEAN_PENALTY_SCALE * mean_pen 
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
        with open(f"{H.OUT_DIR}/losses/G_loss.txt", "a") as f:
            f.write(f"ITERATION {it:.4f} |  | TOTAL G LOSS {loss_G:.4f}\n")
        #dsisc is same 
        for d_it in range(H.DISC_STEPS):
            try:
                real_batch, = next(real_iter)
            except StopIteration:
                real_iter = iter(loader)
                real_batch, = next(real_iter)
            real_batch = real_batch.to(H.DEVICE)
            with torch.no_grad():
                fake_batch, _, _, _ = G.sample(torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE))
            fake_batch = fake_batch.detach()
            real_batch.requires_grad_(True)
            real_logits = D(real_batch)
            grad_real = torch.autograd.grad(real_logits.sum(), real_batch, create_graph=True)[0]
            gp = H.GRADIENT_PENALTY * 0.5 * grad_real.pow(2).view(real_batch.size(0), -1).sum(1).mean()
            loss_D = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_batch[:, :1])) + F.binary_cross_entropy_with_logits(D(fake_batch), torch.zeros_like(fake_batch[:, :1])) + gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
        if it % 50 == 0:
            print(f"{it} complete")
            with open(f"{H.OUT_DIR}/losses/output.txt", "a") as f:
                f.write(f"iteration {it} | D LOSS = {loss_D.item():.4f} | G LOSS = {loss_G.item():.4f} | mean_pen: {(mean_pen.item()*H.MEAN_PENALTY_SCALE):.4f} |  \n")
    #generate and save  
    z = torch.randn(H.NUM_SAMPLES, H.NOISE_DIM, device=H.DEVICE)
    synthetic, _, _, _ = G.sample(z)
    cols = H.NUM_COLS + H.CAT_COLS
    if H.DEVICE == "cuda": 
        df_syn = pd.DataFrame(synthetic.cpu().detach().numpy(), columns=cols)
    else: 
        df_syn = pd.DataFrame(synthetic.detach().numpy(), columns=cols)
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic.csv")
    #rescale 
    feature_range = np.load(H.NPY_PATH, allow_pickle=True).item()
    for col in H.NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")
    elapsed_time = (time.time() - start_time) / 60
    return df_syn, elapsed_time


def train(df_train, real, loader, H): 
    start_time = time.time() 
    #instantiate
    G, D = build_models(H)
    opt_G = torch.optim.Adam(G.parameters(), lr=H.G_LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=H.D_LR)
    os.makedirs(f"{H.OUT_DIR}/losses", exist_ok=True)
    with open(f"{H.OUT_DIR}/losses/output.txt", "w") as f:
        f.write(f"Logging\n")
    with open(f"{H.OUT_DIR}/losses/G_loss.txt", "w") as f:
        f.write(f"Logging\n")
    #ppo training loop
    real_iter = iter(loader)
    for it in range(H.ITERS):
        #rollout policy 
        z = torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE)
        rows, logp_old, _, v_old = G.sample(z)
        #detach for grad
        rows = rows.detach()      
        logp_old = logp_old.detach()
        v_old = v_old.detach()
        with torch.no_grad():
            rewards = torch.sigmoid(D(rows)).squeeze()
        adv = rewards - v_old 
        #normalize to smooth advantage 
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        #mean penalty 
        with torch.no_grad():
            target_mean = real.mean(0, keepdim=True).to(H.DEVICE)
        num_fake = rows[:, :len(H.NUM_COLS)]
        mean_pen = (num_fake.mean(0) - target_mean[0, :len(H.NUM_COLS)]).pow(2).mean()
        #PPO update 
        for _ in range(H.PPO_EPOCHS):
            logp, v = G.eval_action(z, rows)
            ratio = (logp - logp_old).exp()
            surr1 = ratio * adv_n
            surr2 = torch.clamp(ratio, 1-H.CLIP_EPS, 1+H.CLIP_EPS) * adv_n 
            loss_pi = -(torch.min(surr1, surr2)).mean()
            loss_v = F.mse_loss(v, rewards)
            entropy = -logp.mean()
            loss_G = loss_pi + H.VF_COEF * loss_v - H.ENT_BETA * entropy
            loss_G += H.MEAN_PENALTY_SCALE * mean_pen 
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            with open(f"{H.OUT_DIR}/losses/G_loss.txt", "a") as f:
                f.write(f"ITERATION {it:.4f} | LOSS PI {loss_pi:.4f} | LOSS V {H.VF_COEF*loss_v:.4f} | ENTROPY {(H.ENT_BETA*entropy):.4f} | MEAN PEN (*.2) {(H.MEAN_PENALTY_SCALE * mean_pen):.4f} | TOTAL G LOSS {loss_G:.4f}\n")
        #discriminator update 
        for d_it in range(H.DISC_STEPS):
            try:
                real_batch, = next(real_iter)
            except StopIteration:
                real_iter = iter(loader)
                real_batch, = next(real_iter)
            real_batch = real_batch.to(H.DEVICE)
            #fresh fake batch
            fake_batch, _, _, _ = G.sample(torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE))
            fake_batch = fake_batch.detach()
            #R1 gradient penalty on real data --> not quite WGAN!!! Could try with wasserstein distance 
            real_batch.requires_grad_(True)
            real_logits = D(real_batch)
            grad_real = torch.autograd.grad(real_logits.sum(), real_batch, create_graph=True)[0]
            gp = H.GRADIENT_PENALTY * 0.5 * grad_real.pow(2).view(real_batch.size(0), -1).sum(1).mean()
            loss_D = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_batch[:, :1])) + F.binary_cross_entropy_with_logits(D(fake_batch), torch.zeros_like(fake_batch[:, :1])) + gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
        if it % 50 == 0:
            print(f"{it} complete")
            with open(f"{H.OUT_DIR}/losses/output.txt", "a") as f:
                f.write(f"iteration {it} | D LOSS = {loss_D.item():.4f} | G LOSS = {loss_G.item():.4f} | AVG R ={rewards.mean():.4f} | mean_pen: {(mean_pen.item()*H.MEAN_PENALTY_SCALE):.4f} |  \n")
    #generate and save  
    z = torch.randn(H.NUM_SAMPLES, H.NOISE_DIM, device=H.DEVICE)
    synthetic, _, _, _ = G.sample(z)
    cols = H.NUM_COLS + H.CAT_COLS
    if H.DEVICE == "cuda": 
        df_syn = pd.DataFrame(synthetic.cpu().detach().numpy(), columns=cols)
    else: 
        df_syn = pd.DataFrame(synthetic.detach().numpy(), columns=cols)
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic.csv")
    #rescale 
    feature_range = np.load(H.NPY_PATH, allow_pickle=True).item()
    for col in H.NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")
    elapsed_time = (time.time() - start_time) / 60 # minutes 
    return df_syn, elapsed_time

def det_GAN(df_train, real, loader, H):
    start_time = time.time()

    class SimpleGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            h = H.G_H
            self.net = nn.Sequential(
                nn.Linear(H.NOISE_DIM, h), nn.ReLU(),
                nn.Linear(h, h), nn.ReLU(),
                nn.Linear(h, len(H.NUM_COLS) + H.CAT_DIM)
            )
        def forward(self, z):
            out = self.net(z)
            num = torch.sigmoid(out[:, :len(H.NUM_COLS)])
            cat = torch.sigmoid(out[:, len(H.NUM_COLS):])
            return torch.cat([num, cat], 1)

    class Disc(nn.Module):
        def __init__(self):
            super().__init__()
            h = H.D_H
            self.fc = nn.Sequential(
                nn.Linear(len(H.NUM_COLS) + H.CAT_DIM, h), nn.LeakyReLU(0.2),
                nn.Linear(h, h), nn.LeakyReLU(0.2),
                nn.Linear(h, 1)
            )
        def forward(self, row):
            return self.fc(row)

    G = SimpleGenerator().to(H.DEVICE)
    D = Disc().to(H.DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=H.G_LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=H.D_LR)
    os.makedirs(f"{H.OUT_DIR}/losses", exist_ok=True)
    with open(f"{H.OUT_DIR}/losses/output.txt", "w") as f:
        f.write(f"Logging\n")

    real_iter = iter(loader)
    for it in range(H.ITERS):
        z = torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE)
        rows = G(z)
        fake_logits = D(rows)
        loss_G = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        for d_it in range(H.DISC_STEPS):
            try:
                real_batch, = next(real_iter)
            except StopIteration:
                real_iter = iter(loader)
                real_batch, = next(real_iter)
            real_batch = real_batch.to(H.DEVICE)
            with torch.no_grad():
                fake_batch = G(torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE))
            real_batch.requires_grad_(True)
            real_logits = D(real_batch)
            grad_real = torch.autograd.grad(real_logits.sum(), real_batch, create_graph=True)[0]
            gp = H.GRADIENT_PENALTY * 0.5 * grad_real.pow(2).view(real_batch.size(0), -1).sum(1).mean()
            loss_D = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_batch[:, :1]))
                    + F.binary_cross_entropy_with_logits(D(fake_batch), torch.zeros_like(fake_batch[:, :1]))
                    + gp)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        if it % 5000 == 0:
            print(f"{it} complete")
            with open(f"{H.OUT_DIR}/losses/output.txt", "a") as f:
                f.write(f"iteration {it} | D LOSS = {loss_D.item():.4f} | G LOSS = {loss_G.item():.4f}\n")

    with torch.no_grad():
        synthetic = G(torch.randn(H.NUM_SAMPLES, H.NOISE_DIM, device=H.DEVICE))
    cols = H.NUM_COLS + H.CAT_COLS
    if H.DEVICE == "cuda":
        df_syn = pd.DataFrame(synthetic.cpu().numpy(), columns=cols)
    else:
        df_syn = pd.DataFrame(synthetic.numpy(), columns=cols)
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic.csv")
    feature_range = np.load(H.NPY_PATH, allow_pickle=True).item()
    for col in H.NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")
    elapsed_time = (time.time() - start_time) / 60
    return df_syn, elapsed_time

def no_clip_no_value(df_train, real, loader, H): 
    start_time = time.time() 
    #instantiate
    G, D = build_models(H)
    opt_G = torch.optim.Adam(G.parameters(), lr=H.G_LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=H.D_LR)
    os.makedirs(f"{H.OUT_DIR}/losses", exist_ok=True)
    with open(f"{H.OUT_DIR}/losses/output.txt", "w") as f:
        f.write(f"Logging\n")
    with open(f"{H.OUT_DIR}/losses/G_loss.txt", "w") as f:
        f.write(f"Logging\n")
    #ppo training loop
    real_iter = iter(loader)
    for it in range(H.ITERS):
        #rollout policy 
        z = torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE)
        rows, logp_old, _, v_old = G.sample(z)
        #detach for grad
        rows = rows.detach()      
        logp_old = logp_old.detach()
        v_old = v_old.detach()
        with torch.no_grad():
            rewards = torch.sigmoid(D(rows)).squeeze()
        adv_n = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        #mean penalty 
        with torch.no_grad():
            target_mean = real.mean(0, keepdim=True).to(H.DEVICE)
        num_fake = rows[:, :len(H.NUM_COLS)]
        mean_pen = (num_fake.mean(0) - target_mean[0, :len(H.NUM_COLS)]).pow(2).mean()
        #PPO update 
        for _ in range(H.PPO_EPOCHS):
            logp, v = G.eval_action(z, rows)
            ratio = (logp - logp_old).exp()
            #surr1 = ratio * adv_n
            #surr2 = torch.clamp(ratio, 1-H.CLIP_EPS, 1+H.CLIP_EPS) * adv_n 
            loss_pi = -(ratio * adv_n).mean()
            #loss_pi = -(torch.min(surr1, surr2)).mean()
            loss_v = F.mse_loss(v, rewards)
            entropy = -logp.mean()
            loss_G = loss_pi - H.ENT_BETA * entropy
            loss_G += H.MEAN_PENALTY_SCALE * mean_pen 
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            with open(f"{H.OUT_DIR}/losses/G_loss.txt", "a") as f:
                f.write(f"ITERATION {it:.4f} | LOSS PI {loss_pi:.4f} | LOSS V {H.VF_COEF*loss_v:.4f} | ENTROPY {(H.ENT_BETA*entropy):.4f} | MEAN PEN (*.2) {(H.MEAN_PENALTY_SCALE * mean_pen):.4f} | TOTAL G LOSS {loss_G:.4f}\n")
        #discriminator update 
        for d_it in range(H.DISC_STEPS):
            try:
                real_batch, = next(real_iter)
            except StopIteration:
                real_iter = iter(loader)
                real_batch, = next(real_iter)
            real_batch = real_batch.to(H.DEVICE)
            #fresh fake batch
            fake_batch, _, _, _ = G.sample(torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE))
            fake_batch = fake_batch.detach()
            #R1 gradient penalty on real data --> not quite WGAN!!! Could try with wasserstein distance 
            real_batch.requires_grad_(True)
            real_logits = D(real_batch)
            grad_real = torch.autograd.grad(real_logits.sum(), real_batch, create_graph=True)[0]
            gp = H.GRADIENT_PENALTY * 0.5 * grad_real.pow(2).view(real_batch.size(0), -1).sum(1).mean()
            loss_D = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_batch[:, :1])) + F.binary_cross_entropy_with_logits(D(fake_batch), torch.zeros_like(fake_batch[:, :1])) + gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
        if it % 50 == 0:
            print(f"{it} complete")
            with open(f"{H.OUT_DIR}/losses/output.txt", "a") as f:
                f.write(f"iteration {it} | D LOSS = {loss_D.item():.4f} | G LOSS = {loss_G.item():.4f} | AVG R ={rewards.mean():.4f} | mean_pen: {(mean_pen.item()*H.MEAN_PENALTY_SCALE):.4f} |  \n")
    #generate and save  
    z = torch.randn(H.NUM_SAMPLES, H.NOISE_DIM, device=H.DEVICE)
    synthetic, _, _, _ = G.sample(z)
    cols = H.NUM_COLS + H.CAT_COLS
    if H.DEVICE == "cuda": 
        df_syn = pd.DataFrame(synthetic.cpu().detach().numpy(), columns=cols)
    else: 
        df_syn = pd.DataFrame(synthetic.detach().numpy(), columns=cols)
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic.csv")
    #rescale 
    feature_range = np.load(H.NPY_PATH, allow_pickle=True).item()
    for col in H.NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")
    elapsed_time = (time.time() - start_time) / 60 # minutes 
    return df_syn, elapsed_time



def no_clip(df_train, real, loader, H): 
    start_time = time.time() 
    #instantiate
    G, D = build_models(H)
    opt_G = torch.optim.Adam(G.parameters(), lr=H.G_LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=H.D_LR)
    os.makedirs(f"{H.OUT_DIR}/losses", exist_ok=True)
    with open(f"{H.OUT_DIR}/losses/output.txt", "w") as f:
        f.write(f"Logging\n")
    with open(f"{H.OUT_DIR}/losses/G_loss.txt", "w") as f:
        f.write(f"Logging\n")
    #ppo training loop
    real_iter = iter(loader)
    for it in range(H.ITERS):
        #rollout policy 
        z = torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE)
        rows, logp_old, _, v_old = G.sample(z)
        #detach for grad
        rows = rows.detach()      
        logp_old = logp_old.detach()
        v_old = v_old.detach()
        with torch.no_grad():
            rewards = torch.sigmoid(D(rows)).squeeze()
        adv = rewards - v_old 
        #normalize to smooth advantage 
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        #mean penalty 
        with torch.no_grad():
            target_mean = real.mean(0, keepdim=True).to(H.DEVICE)
        num_fake = rows[:, :len(H.NUM_COLS)]
        mean_pen = (num_fake.mean(0) - target_mean[0, :len(H.NUM_COLS)]).pow(2).mean()
        #PPO update 
        for _ in range(H.PPO_EPOCHS):
            logp, v = G.eval_action(z, rows)
            ratio = (logp - logp_old).exp()
            #surr1 = ratio * adv_n
            #surr2 = torch.clamp(ratio, 1-H.CLIP_EPS, 1+H.CLIP_EPS) * adv_n 
            loss_pi = -(ratio * adv_n).mean()
            #loss_pi = -(torch.min(surr1, surr2)).mean()
            loss_v = F.mse_loss(v, rewards)
            entropy = -logp.mean()
            loss_G = loss_pi + H.VF_COEF * loss_v - H.ENT_BETA * entropy
            loss_G += H.MEAN_PENALTY_SCALE * mean_pen 
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            with open(f"{H.OUT_DIR}/losses/G_loss.txt", "a") as f:
                f.write(f"ITERATION {it:.4f} | LOSS PI {loss_pi:.4f} | LOSS V {H.VF_COEF*loss_v:.4f} | ENTROPY {(H.ENT_BETA*entropy):.4f} | MEAN PEN (*.2) {(H.MEAN_PENALTY_SCALE * mean_pen):.4f} | TOTAL G LOSS {loss_G:.4f}\n")
        #discriminator update 
        for d_it in range(H.DISC_STEPS):
            try:
                real_batch, = next(real_iter)
            except StopIteration:
                real_iter = iter(loader)
                real_batch, = next(real_iter)
            real_batch = real_batch.to(H.DEVICE)
            #fresh fake batch
            fake_batch, _, _, _ = G.sample(torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE))
            fake_batch = fake_batch.detach()
            #R1 gradient penalty on real data --> not quite WGAN!!! Could try with wasserstein distance 
            real_batch.requires_grad_(True)
            real_logits = D(real_batch)
            grad_real = torch.autograd.grad(real_logits.sum(), real_batch, create_graph=True)[0]
            gp = H.GRADIENT_PENALTY * 0.5 * grad_real.pow(2).view(real_batch.size(0), -1).sum(1).mean()
            loss_D = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_batch[:, :1])) + F.binary_cross_entropy_with_logits(D(fake_batch), torch.zeros_like(fake_batch[:, :1])) + gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
        if it % 50 == 0:
            print(f"{it} complete")
            with open(f"{H.OUT_DIR}/losses/output.txt", "a") as f:
                f.write(f"iteration {it} | D LOSS = {loss_D.item():.4f} | G LOSS = {loss_G.item():.4f} | AVG R ={rewards.mean():.4f} | mean_pen: {(mean_pen.item()*H.MEAN_PENALTY_SCALE):.4f} |  \n")
    #generate and save  
    z = torch.randn(H.NUM_SAMPLES, H.NOISE_DIM, device=H.DEVICE)
    synthetic, _, _, _ = G.sample(z)
    cols = H.NUM_COLS + H.CAT_COLS
    if H.DEVICE == "cuda": 
        df_syn = pd.DataFrame(synthetic.cpu().detach().numpy(), columns=cols)
    else: 
        df_syn = pd.DataFrame(synthetic.detach().numpy(), columns=cols)
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic.csv")
    #rescale 
    feature_range = np.load(H.NPY_PATH, allow_pickle=True).item()
    for col in H.NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")
    elapsed_time = (time.time() - start_time) / 60 # minutes 
    return df_syn, elapsed_time

def no_value(df_train, real, loader, H): 
    start_time = time.time() 
    #instantiate
    G, D = build_models(H)
    opt_G = torch.optim.Adam(G.parameters(), lr=H.G_LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=H.D_LR)
    os.makedirs(f"{H.OUT_DIR}/losses", exist_ok=True)
    with open(f"{H.OUT_DIR}/losses/output.txt", "w") as f:
        f.write(f"Logging\n")
    with open(f"{H.OUT_DIR}/losses/G_loss.txt", "w") as f:
        f.write(f"Logging\n")
    #ppo training loop
    real_iter = iter(loader)
    for it in range(H.ITERS):
        #rollout policy 
        z = torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE)
        rows, logp_old, _, v_old = G.sample(z)
        #detach for grad
        rows = rows.detach()      
        logp_old = logp_old.detach()
        v_old = v_old.detach()
        with torch.no_grad():
            rewards = torch.sigmoid(D(rows)).squeeze()
        #NO VALUE
        adv_n = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        #mean penalty 
        with torch.no_grad():
            target_mean = real.mean(0, keepdim=True).to(H.DEVICE)
        num_fake = rows[:, :len(H.NUM_COLS)]
        mean_pen = (num_fake.mean(0) - target_mean[0, :len(H.NUM_COLS)]).pow(2).mean()
        #PPO update 
        for _ in range(H.PPO_EPOCHS):
            logp, v = G.eval_action(z, rows)
            ratio = (logp - logp_old).exp()
            surr1 = ratio * adv_n
            surr2 = torch.clamp(ratio, 1-H.CLIP_EPS, 1+H.CLIP_EPS) * adv_n 
            loss_pi = -(torch.min(surr1, surr2)).mean()
            entropy = -logp.mean()
            loss_G = loss_pi - H.ENT_BETA * entropy
            loss_G += H.MEAN_PENALTY_SCALE * mean_pen 
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            with open(f"{H.OUT_DIR}/losses/G_loss.txt", "a") as f:
                f.write(f"ITERATION {it:.4f} | LOSS PI {loss_pi:.4f}  ENTROPY {(H.ENT_BETA*entropy):.4f} | MEAN PEN (*.2) {(H.MEAN_PENALTY_SCALE * mean_pen):.4f} | TOTAL G LOSS {loss_G:.4f}\n")
        #discriminator update 
        for d_it in range(H.DISC_STEPS):
            try:
                real_batch, = next(real_iter)
            except StopIteration:
                real_iter = iter(loader)
                real_batch, = next(real_iter)
            real_batch = real_batch.to(H.DEVICE)
            #fresh fake batch
            fake_batch, _, _, _ = G.sample(torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE))
            fake_batch = fake_batch.detach()
            #R1 gradient penalty on real data --> not quite WGAN!!! Could try with wasserstein distance 
            real_batch.requires_grad_(True)
            real_logits = D(real_batch)
            grad_real = torch.autograd.grad(real_logits.sum(), real_batch, create_graph=True)[0]
            gp = H.GRADIENT_PENALTY * 0.5 * grad_real.pow(2).view(real_batch.size(0), -1).sum(1).mean()
            loss_D = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_batch[:, :1])) + F.binary_cross_entropy_with_logits(D(fake_batch), torch.zeros_like(fake_batch[:, :1])) + gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
        if it % 50 == 0:
            print(f"{it} complete")
            with open(f"{H.OUT_DIR}/losses/output.txt", "a") as f:
                f.write(f"iteration {it} | D LOSS = {loss_D.item():.4f} | G LOSS = {loss_G.item():.4f} | AVG R ={rewards.mean():.4f} | mean_pen: {(mean_pen.item()*H.MEAN_PENALTY_SCALE):.4f} |  \n")
    #generate and save  
    z = torch.randn(H.NUM_SAMPLES, H.NOISE_DIM, device=H.DEVICE)
    synthetic, _, _, _ = G.sample(z)
    cols = H.NUM_COLS + H.CAT_COLS
    if H.DEVICE == "cuda": 
        df_syn = pd.DataFrame(synthetic.cpu().detach().numpy(), columns=cols)
    else: 
        df_syn = pd.DataFrame(synthetic.detach().numpy(), columns=cols)
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic.csv")
    #rescale 
    feature_range = np.load(H.NPY_PATH, allow_pickle=True).item()
    for col in H.NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")
    elapsed_time = (time.time() - start_time) / 60 # minutes 
    return df_syn, elapsed_time

def evaluate_model_aireadi(df_real, df_hold, df_train, df_syn, df_hold_norm, df_train_norm, df_syn_norm, df_real_with_patients_norm, H, elapsed_time): 
    #plot losses
    plot_rl_losses(f"{H.OUT_DIR}")
    #-----------------FIDELITY-----------------#
    #examine value statistics per feature
    print_stats(f"{H.OUT_DIR}/eval.txt", df_real, df_syn, df_hold, df_train, H.NUM_COLS) 
    #get single feature histograms
    get_histograms(df_real, df_syn, f"{H.OUT_DIR}/histograms")
    #PCA analysis
    latent_cluster_analysis(df_real_with_patients_norm.drop(columns=['patient_id']), [df_syn_norm], f"{H.OUT_DIR}/PCA", H.SEED)
    #column wise correlations 
    cwc = get_column_wise_correlations(df_real, df_syn, f"{H.OUT_DIR}/correlations") 
    #distributional scores by value stats 
    overall_score_real_num, overall_score_hold_num, overall_score_real_cat, overall_score_hold_cat, overall_score_real, overall_score_hold = compute_value_stats(df_real, df_hold, df_syn, f"{H.OUT_DIR}/value_stat_analysis.txt", H.NUM_COLS, H.CAT_COLS)  
    #-----------------UTILITY-----------------#
    #get hold out (keep same size no matter the train - test split)
    ten_percent = 490
    if len(df_hold_norm) > ten_percent: 
        hold_fraction = ten_percent / len(df_hold_norm)   
        df_hold_norm = df_hold_norm.sample(frac=hold_fraction, random_state=H.SEED).reset_index(drop=True)
    #synthetic to hold out 
    s2h_auc, s2h_acc = train_on_synth_test_on_hold(df_syn_norm,  df_hold_norm, f"{H.OUT_DIR}/synth_to_hold", H.SEED)
    #real to synthetic 
    r2s_auc, r2s_acc = train_on_real_test_on_synth(df_real_with_patients_norm, df_syn_norm, f"{H.OUT_DIR}/real_to_synth", H.SEED)
    #synthetic to real 
    s2r_auc, s2r_acc = train_on_synth_test_on_real(df_syn_norm, df_real_with_patients_norm, f"{H.OUT_DIR}/synth_to_real", H.SEED)
    #real to real 
    r2r_auc, r2r_acc = train_on_real_test_on_real(f"{H.OUT_DIR}/real_to_real", df_real_with_patients_norm, H.SEED)
    #classify real vs syn 
    df_real = df_real_with_patients_norm.copy().drop(columns=["patient_id"])
    df_real["is_syn"] = 0
    df_syn = df_syn_norm.copy()
    df_syn["is_syn"] = 1 
    fraction = len(df_real) / len(df_syn)
    if fraction < 1: 
        df_syn = df_syn.sample(frac=fraction, random_state=H.SEED).reset_index(drop=True)
    elif fraction > 1: 
        new_frac = len(df_syn) / len(df_real) 
        df_real =  df_real.sample(frac=new_frac, random_state=H.SEED).reset_index(drop=True)
    df_all = pd.concat([df_real, df_syn], ignore_index=True)
    rvs_auc, rvs_acc = classify_real_vs_syn(df_all, f"{H.OUT_DIR}/real_vs_syn", H.SEED)
    #-----------------PRIVACY---------------------#
    mem_aucs = mem_risk(df_train_norm, df_hold_norm, df_syn_norm, H.CAT_COLS, H.NUM_COLS, f"{H.OUT_DIR}/mem_risk", H.SEED) 
    #-----------------LOG RESULTS-----------------#
    with open(f"{H.OUT_DIR}/eval.txt", "a") as f:
        f.write(f"CWC: {cwc}\noverall_score_hold : {overall_score_hold}\noverall_score_real: {overall_score_real}\noverall_score_real_num: {overall_score_real_num}\noverall_score_hold_num: {overall_score_hold_num}\noverall_score_real_cat: {overall_score_real_cat}\noverall_score_hold_cat:{overall_score_hold_cat}\n"
            f"r2r_auc: {r2r_auc}\nr2r_acc: {r2r_acc}\nr2s_auc: {r2s_auc}\nr2s_acc: {r2s_acc}\ns2r_auc: {s2r_auc}\ns2r_acc: {s2r_acc}\ns2h_auc: {s2h_auc}\ns2h_acc: {s2h_acc}\nrvs_auc: {rvs_auc}\nrvs_acc: {rvs_acc}\n"
            f"real_mem_auc_: {mem_aucs['real_roc_auc']}\nreal_mem_auc_bal: {mem_aucs['real_roc_auc_bal']}\nsynth_mem_auc: {mem_aucs['synth_roc_auc']}\nsynth_mem_auc_bal: {mem_aucs['synth_roc_auc_bal']}\nelapsed_time: {elapsed_time}")
    log_result_RL(H.RESULT_CSV, H.RUN_NAME, H.ITERS, H.DATA_SIZE, H.SEED, cwc, overall_score_real_num, overall_score_hold_num,  overall_score_real_cat, overall_score_hold_cat, overall_score_real, overall_score_hold, 
            r2r_auc, r2r_acc, s2h_auc, s2h_acc, s2r_auc, s2r_acc, r2s_auc, r2s_acc, rvs_auc, rvs_acc, mem_aucs["real_roc_auc"], mem_aucs["real_roc_auc_bal"], mem_aucs["synth_roc_auc"], mem_aucs["synth_roc_auc_bal"], elapsed_time)


def evaluate_model_MIMIC(df_train, df_real, df_syn, df_train_norm, df_hold_norm, df_real_norm, df_syn_norm, H, elapsed_time): 
    #-----------------FIDELITY-----------------#
    #column wise correlations 
    cwc = get_column_wise_correlationsM(df_real, df_syn, f"{H.OUT_DIR}/correlations", True) 
    ad2d, continuous_w_d = compute_dimension_wide_distribution(df_real, df_syn, f"{H.OUT_DIR}/dimension_wide_distributions")
    latent_cluster_analysis = latent_cluster_analysisM(df_real, df_syn, f"{H.OUT_DIR}/PCA")
    #run_PCA(df_real, [df_syn], f"{H.OUT_DIR}/PCA", H.SEED)
    mca_dist, mca_tvd_dist = medical_concept_abundance(df_real, df_syn, H.CAT_COLS, f"{H.OUT_DIR}/medical_abundance")
    combined_clinical_violations = clinical_knowledge_violation(df_train, df_syn, H.CAT_COLS, f"{H.OUT_DIR}/clinical_knowledge_violation")
    EXCLUDE_COLS = ['WHITE', 'BLACK', 'ASIAN', 'HISPANIC', 'UN', 'OTHER', 'DIE_1y']
    cat_cols = [c for c in H.CAT_COLS if c not in EXCLUDE_COLS]
    r2s_results = train_and_test_classification(df_real_norm, df_syn_norm, f"{H.OUT_DIR}/real_to_synthetic", H.SEED, cat_cols)
    r2s_auc = r2s_results["test"]["auroc"]
    r2s_prauc = r2s_results["test"]["prauc"]
    r2s_acc  = r2s_results["test"]["acc"]
    #-----------------UTILITY-----------------#
    s2h_results = train_and_test_classification(df_syn_norm, df_hold_norm, f"{H.OUT_DIR}/synthetic_to_hold", H.SEED,  cat_cols)
    s2h_auc = s2h_results["test"]["auroc"]
    s2h_prauc = s2h_results["test"]["prauc"]
    s2h_acc  = s2h_results["test"]["acc"]
    #r2r_results = train_and_test_classification(df_train_norm, df_hold_norm, f"{H.OUT_DIR}/real_to_real", H.SEED, cat_cols)
    r2r_auc = 0 # r2r_results["test"]["auroc"]
    r2r_prauc = 0 # r2r_results["test"]["prauc"]
    r2r_acc  = 0 # r2r_results["test"]["acc"]
    #-----------------PRIVACY-----------------#
    #df_train_norm_bal = df_train_norm.sample(n=30000, random_state=H.SEED)[H.NUM_COLS + H.CAT_COLS]
    #df_hold_norm_bal = df_hold_norm.sample(n=30000, random_state=H.SEED)[H.NUM_COLS + H.CAT_COLS]
    df_train_norm_unbal = df_train_norm.sample(n=21000, random_state=H.SEED)[H.NUM_COLS + H.CAT_COLS]
    df_hold_norm_unbal = df_hold_norm.sample(n=9000, random_state=H.SEED)[H.NUM_COLS + H.CAT_COLS]
    mem_aucs = mem_risk_MIMIC(df_train_norm_unbal, df_hold_norm_unbal, df_syn_norm, H.CAT_COLS, H.NUM_COLS, f"{H.OUT_DIR}/mem_risk", H.SEED)
    #-----------------LOG RESULTS-----------------#
    with open(f"{H.OUT_DIR}/eval.txt", "a") as f:
        f.write(f"CWC: {cwc}\nr2s_auc: {r2s_auc}\nr2s_prauc: {r2s_prauc}\nr2s_acc: {r2s_acc}\ns2h_auc: {s2h_auc}\ns2h_prauc: {s2h_prauc}\ns2h_acc: {s2h_acc}\n"
                f"r2r_auc: {r2r_auc}\nr2r_prauc: {r2r_prauc}\nr2r_acc: {r2r_acc}\n"
                f"ad2d: {ad2d}\ncontinuous_w_d: {continuous_w_d}\nlatent_cluster_analysis: {latent_cluster_analysis}\nmca_dist: {mca_dist}\nmca_tvd_dist: {mca_tvd_dist}\ncombined_clinical_violations: {combined_clinical_violations}\nmem_auc_real: {mem_aucs['real_roc_auc']}\nmem_auc_synth: {mem_aucs['synth_roc_auc']}\n elapsed_time: {elapsed_time}\n")
    log_result_RL_MIMIC(H.RESULT_CSV, H.RUN_NAME, H.ITERS, H.DATA_SIZE, H.SEED, cwc, ad2d, continuous_w_d, latent_cluster_analysis, mca_dist, mca_tvd_dist, combined_clinical_violations, s2h_auc, s2h_prauc, s2h_acc, r2s_auc, r2s_prauc, r2s_acc, r2r_auc, r2r_prauc, r2r_acc, mem_aucs, elapsed_time)
    return s2h_auc, r2s_auc 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--OUT_DIR", type=str, required=True )
    p.add_argument("--RESULT_CSV", type=str, required=True)
    p.add_argument("--RUN_NAME", type=str, required=True)
    p.add_argument("--DATASET", choices=["AIREADI", "MIMIC"], required=True)
    p.add_argument("--DATA_PATH", type=str, required=True)
    p.add_argument("--SEED", type=int, required=True)
    p.add_argument("--ABLATION", type=str)
    #add more here if you want to override other hps. 
    return vars(p.parse_args())


def main(): 
    global H 
    args = parse_args() 
    if args["DATASET"] == "AIREADI": 
        H = HyperParams_AIREADI().override(**args)   
        H = H.override(
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu", 
            NPY_PATH = f"{H.DATA_PATH}/min_max_log.npy", 

        )
    elif args["DATASET"] == "MIMIC": 
        H = HyperParams_MIMIC().override(**args)   
        H = H.override( 
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu", 
            NPY_PATH = f"MIMIC_DATA/min_max_log.npy",
            #NPY_PATH = f"RLSYN_paper_results/MIMIC/MIMIC_DATA/seeds/min_max_log.npy",
        )
    #train
    set_seed(H.SEED)
    os.makedirs(H.OUT_DIR, exist_ok=True)
    if H.DATASET == "MIMIC": 
        df_train = pd.read_csv(f"{H.DATA_PATH}/normalized_training_data_{H.SEED}.csv")[H.NUM_COLS + H.CAT_COLS]
    else: 
        df_train = pd.read_csv(f"{H.DATA_PATH}/normalized_training_data.csv")[H.NUM_COLS + H.CAT_COLS]
    real = torch.tensor(df_train.values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(real), batch_size=H.BATCH, shuffle=True, num_workers=0) 

    #ablations

    '''
    6,[0.8830357142857143, 0.8590352811215688, 0.5201030556666109], 
    {'batch': 256, 'noise_dim': 112, 'ppo_epochs': 5, 'disc_steps': 3, 'mean_penalty': 0.2, 'gradient_penalty': 5, 'g_lr': 0.0001, 'd_lr': 5e-05}

    '''
    if H.ABLATION == "trial6_ogoptuna": 
        H = H.override(
            BATCH = 256, 
            NOISE_DIM = 112, 
            PPO_EPOCHS = 5, 
            DISC_STEPS = 3, 
            MEAN_PENALTY_SCALE = 0.2, 
            GRADIENT_PENALTY = 5, 
            G_LR = 0.0001, 
            D_LR = 5e-5, 
            G_H = 64, 
            D_H = 64, 
        )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    
    if H.ABLATION == "cwc_trial2": 
        H = H.override(
            BATCH = 256, DISC_STEPS =5,D_LR =  6.156391413543428e-05, G_LR =  6.156391413543428e-05, PPO_EPOCHS = 4,
            NOISE_DIM = 64, GRADIENT_PENALTY=10, G_H = 128, D_H = 128,  MEAN_PENALTY_SCALE=0.2, USE_TANH=True,   )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    if H.ABLATION == "cwc_trial14": 
        H = H.override(
            BATCH = 384, DISC_STEPS =4,D_LR =  7.02679030270544e-05, G_LR =  7.02679030270544e-05, PPO_EPOCHS = 3,
            NOISE_DIM = 64, GRADIENT_PENALTY=10, G_H = 128, D_H = 128,  MEAN_PENALTY_SCALE=0.2, USE_TANH=True,   )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    if H.ABLATION == "cwc_trial25": 
        H = H.override(
            BATCH = 128, DISC_STEPS =5, D_LR =  7.883987180531893e-05, G_LR =  7.883987180531893e-05, PPO_EPOCHS = 5,
            NOISE_DIM = 64, GRADIENT_PENALTY=10, G_H = 128, D_H = 128,  MEAN_PENALTY_SCALE=0.2, USE_TANH=True,   )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    
    if H.ABLATION == "optuna3_trial2": 
        H = H.override(BATCH = 384, NOISE_DIM = 64, DISC_STEPS =3, GRADIENT_PENALTY=10,
        D_LR = 3e-5, G_LR = 5e-5, G_H = 64, D_H = 64, PPO_EPOCHS = 5, MEAN_PENALTY_SCALE=0, USE_TANH=True,   )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    if H.ABLATION == "optuna3_trial12": 
        H = H.override(BATCH = 128, NOISE_DIM = 64, DISC_STEPS =3, GRADIENT_PENALTY=10,
        D_LR = 5e-5, G_LR = 0.0002, G_H = 64, D_H = 64, PPO_EPOCHS = 3, MEAN_PENALTY_SCALE=0, USE_TANH=True,   )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    if H.ABLATION == "optuna3_trial9": 
        H = H.override(BATCH = 384, NOISE_DIM = 64, DISC_STEPS =3, GRADIENT_PENALTY=10,
        D_LR = 1e-5, G_LR = 0.0001, G_H = 64, D_H = 64, PPO_EPOCHS = 5, MEAN_PENALTY_SCALE=0, USE_TANH=True,   )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    if H.ABLATION == "optuna3_trial6": 
        H = H.override(BATCH = 128, NOISE_DIM = 64, DISC_STEPS =3, GRADIENT_PENALTY=10,
        D_LR = 3e-5, G_LR = 0.0005, G_H = 64, D_H = 64, PPO_EPOCHS = 3, MEAN_PENALTY_SCALE=0, USE_TANH=True,   )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    

    if H.ABLATION == "optuna2_trial25": 
        H = H.override(BATCH = 256, NOISE_DIM = 112, DISC_STEPS =5, GRADIENT_PENALTY=5,
        D_LR = 1e-5, G_LR = 0.0002, G_H = 128, D_H = 128, PPO_EPOCHS = 3, MEAN_PENALTY_SCALE=0, USE_TANH=True,   )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    if H.ABLATION == "optuna2_trial22": 
        H = H.override(BATCH = 128, NOISE_DIM = 64, DISC_STEPS =5, GRADIENT_PENALTY=5, 
        D_LR = 3e-5, G_LR = 0.00005, G_H = 128, D_H = 64, PPO_EPOCHS = 5, MEAN_PENALTY_SCALE=0, USE_TANH=True,   )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    if H.ABLATION == "optuna2_trial26": 
        H = H.override(BATCH = 256, NOISE_DIM = 64, DISC_STEPS =5, GRADIENT_PENALTY=10, 
        D_LR = 1e-5, G_LR = 0.0002, G_H = 128, D_H = 128, PPO_EPOCHS = 3, MEAN_PENALTY_SCALE=0, USE_TANH=True,   )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    if H.ABLATION == "optuna2_trial5": 
        H = H.override(BATCH = 384, NOISE_DIM = 64, DISC_STEPS =3, GRADIENT_PENALTY=10,
        D_LR = 3e-5, G_LR = 0.00005, G_H = 64, D_H = 64, PPO_EPOCHS = 5, MEAN_PENALTY_SCALE=0.2, USE_TANH=True,   )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    if H.ABLATION == "optuna2_trial9": 
        H = H.override(BATCH = 384, NOISE_DIM = 64, DISC_STEPS =3, GRADIENT_PENALTY=5,
        D_LR = 1e-5, G_LR = 0.0002, G_H = 64, D_H = 64, PPO_EPOCHS = 1, MEAN_PENALTY_SCALE=0.2, USE_TANH=True,   )
        df_syn, elapsed_time = train(df_train, real, loader, H)
    
    elif H.ABLATION == "reinforce": 
        df_syn, elapsed_time = REINFORCE(df_train, real, loader, H)
    elif H.ABLATION == "reparam_gan": 
        #df_syn, elapsed_time = reparam_GAN(df_train, real, loader, H)
        df_syn, elapsed_time = reparam_GAN_cat(df_train, real, loader, H)
    elif H.ABLATION == "no_entropy": 
        H = H.override(ENT_BETA = 0)
        df_syn, elapsed_time = train(df_train, real, loader, H)
    elif H.ABLATION == "no_clip": 
        df_syn, elapsed_time = no_clip(df_train, real, loader, H)
    elif H.ABLATION == "no_clip_no_value": 
        df_syn, elapsed_time = no_clip_no_value(df_train, real, loader, H)
    elif H.ABLATION == "no_clip_ppoepochs1": 
        H = H.override(PPO_EPOCHS=1) 
        df_syn, elapsed_time = no_clip(df_train, real, loader, H)
    elif H.ABLATION == "no_clip_no_value_ppoepochs1": 
        H = H.override(PPO_EPOCHS=1) 
        df_syn, elapsed_time = no_clip_no_value(df_train, real, loader, H)
    elif H.ABLATION == "det_GAN": 
        df_syn, elapsed_time = det_GAN(df_train, real, loader, H)
    else: 
        df_syn, elapsed_time = train(df_train, real, loader, H)

    if H.DATASET == "AIREADI": 
        #get raw to use for cwc, value stat analysis, histograms etc. 
        df_train = pd.read_csv(f"{H.DATA_PATH}/original_training_data.csv")[H.NUM_COLS+ H.CAT_COLS]
        df_hold = pd.read_csv(f"{H.DATA_PATH}/original_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
        df_real = pd.read_csv(f"{H.DATA_PATH}/original_data_with_patients.csv").drop(columns=['patient_id'])[H.NUM_COLS+H.CAT_COLS]
        df_syn = pd.read_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")[H.NUM_COLS+H.CAT_COLS]
        #get normalized data to use for classifications (need patients to split real data without leakage)
        df_hold_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
        df_real_with_patients_norm =  pd.read_csv(f"{H.DATA_PATH}/preprocessed_data_with_patients.csv")[H.NUM_COLS + H.CAT_COLS +["patient_id"]]
        df_syn_norm = pd.read_csv(f"{H.OUT_DIR}/synthetic.csv")[H.NUM_COLS+H.CAT_COLS]
        df_train_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_training_data.csv")[H.NUM_COLS+ H.CAT_COLS]
        evaluate_model_aireadi(df_real, df_hold, df_train, df_syn, df_hold_norm, df_train_norm, df_syn_norm, df_real_with_patients_norm, H, elapsed_time) 
    else: 
        #get synth 
        df_syn_norm = pd.read_csv(f"{H.OUT_DIR}/synthetic.csv")[H.NUM_COLS+H.CAT_COLS]
        df_syn = pd.read_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")[H.NUM_COLS+H.CAT_COLS]
        #get train hold full
        df_train_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_training_data_{H.SEED}.csv")[H.NUM_COLS + H.CAT_COLS]
        df_hold_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_testing_data_{H.SEED}.csv")[H.NUM_COLS + H.CAT_COLS]
        df_real_norm = pd.concat([df_train_norm, df_hold_norm])[H.NUM_COLS + H.CAT_COLS]
        feature_range = np.load(H.NPY_PATH, allow_pickle=True).item()
        for col in H.NUM_COLS:
            xmin, xmax = feature_range[col]
            df_train_norm[col] = (1.0 - df_train_norm[col]) * xmin + df_train_norm[col] * xmax
            df_hold_norm[col] = (1.0 - df_hold_norm[col]) * xmin + df_hold_norm[col] * xmax
        df_train = df_train_norm 
        df_hold = df_hold_norm 
        df_real = pd.concat([df_train, df_hold])[H.NUM_COLS + H.CAT_COLS]
        df_train_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_training_data_{H.SEED}.csv")[H.NUM_COLS + H.CAT_COLS]
        df_hold_norm =  pd.read_csv(f"{H.DATA_PATH}/normalized_testing_data_{H.SEED}.csv")[H.NUM_COLS + H.CAT_COLS]
        df_real_norm = pd.concat([df_train_norm, df_hold_norm])[H.NUM_COLS + H.CAT_COLS]
        print(f"starting eval {H.ABLATION}")
        evaluate_model_MIMIC(df_train, df_real, df_syn, df_train_norm, df_hold_norm, df_real_norm, df_syn_norm, H, elapsed_time) 

if __name__ == '__main__':
    main() 

