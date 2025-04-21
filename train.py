from src.models import Generator, Discriminator, GaussianDiffusion
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

def discriminator_loss(D, real_imgs, fake_imgs, labels):
    real_logits = D(real_imgs, labels)
    fake_logits = D(fake_imgs, labels)
    real_loss = nn.BCEWithLogitsLoss()(real_logits, torch.ones_like(real_logits))
    fake_loss = nn.BCEWithLogitsLoss()(fake_logits, torch.zeros_like(fake_logits))
    return (real_loss + fake_loss) / 2

def generator_loss(D, fake_imgs, labels):
    fake_logits = D(fake_imgs, labels)
    return nn.BCEWithLogitsLoss()(fake_logits, torch.ones_like(fake_logits))

def l1_loss(y_true, y_pred):
    return nn.L1Loss()(y_pred, y_true)

def trainer(train_loader, val_loader, fold, img_channels=1, 
           num_epochs=25, label_dim=3, learning_rate=0.0001, 
           lambda_l1=50, scheduler_factor=0.5, scheduler_patience=5,
           experiment_name="default"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = int(time.time())
    log_path = f"logs/{experiment_name}/fold_{fold+1}_{timestamp}"
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    generator = Generator().to(device)
    discriminator = Discriminator(img_channels, label_dim).to(device)
    diffusion = GaussianDiffusion().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    scheduler_G = ReduceLROnPlateau(
        optimizer_G,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience
    )
    scheduler_D = ReduceLROnPlateau(
        optimizer_D,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience
    )
    
    scaler = torch.amp.GradScaler()
    train_hist = {
        'D_losses': [],
        'G_losses': [],
        'val_losses': [],
        'l1_losses': [],
        'learning_rates_G': [],
        'learning_rates_D': []
    }
    
    best_val_loss = float('inf')
    previous_lr_G = learning_rate
    previous_lr_D = learning_rate
    
    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            pbar.set_description(f"Fold {fold+1} Epoch {epoch+1}/{num_epochs}")
            G_losses, D_losses, val_losses, l1_losses = [], [], [], []

            generator.train()
            discriminator.train()
            
            for labels, imgs in train_loader:
                labels, imgs = labels.to(device), imgs.to(device)
                t = torch.randint(0, diffusion.timesteps, (imgs.size(0),), device=device)
                noise = diffusion.forward_process(imgs, t)

                optimizer_D.zero_grad()
                with torch.amp.autocast("cuda"):
                    fake_images = generator(noise, labels)
                    d_loss = discriminator_loss(discriminator, imgs, fake_images, labels)
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_D)
                scaler.update()

                optimizer_G.zero_grad()
                with torch.amp.autocast("cuda"):
                    fake_images = generator(noise, labels)
                    g_loss = generator_loss(discriminator, fake_images, labels)
                    l1_g_loss_value = l1_loss(imgs, fake_images)
                    total_g_loss = g_loss + lambda_l1 * l1_g_loss_value
                scaler.scale(total_g_loss).backward()
                scaler.step(optimizer_G)
                scaler.update()

                G_losses.append(total_g_loss.item())
                D_losses.append(d_loss.item())
                l1_losses.append(l1_g_loss_value.item())

            generator.eval()
            val_loss = 0
            with torch.no_grad():
                for labels, imgs in val_loader:
                    labels, imgs = labels.to(device), imgs.to(device)
                    t = torch.randint(0, diffusion.timesteps, (imgs.size(0),), device=device)
                    noise = diffusion.forward_process(imgs, t)
                    fake_images = generator(noise, labels)
                    val_loss += generator_loss(discriminator, fake_images, labels).item()
                    val_loss += lambda_l1 * l1_loss(imgs, fake_images).item()
            
            val_loss /= len(val_loader)
            scheduler_G.step(val_loss)
            scheduler_D.step(val_loss)
            
            current_lr_G = optimizer_G.param_groups[0]['lr']
            current_lr_D = optimizer_D.param_groups[0]['lr']
            
            if current_lr_G != previous_lr_G:
                print(f"Generator LR changed from {previous_lr_G:.2e} to {current_lr_G:.2e}")
            if current_lr_D != previous_lr_D:
                print(f"Discriminator LR changed from {previous_lr_D:.2e} to {current_lr_D:.2e}")
            
            previous_lr_G = current_lr_G
            previous_lr_D = current_lr_D
            
            train_hist['D_losses'].append(np.mean(D_losses))
            train_hist['G_losses'].append(np.mean(G_losses))
            train_hist['val_losses'].append(val_loss)
            train_hist['l1_losses'].append(np.mean(l1_losses))
            train_hist['learning_rates_G'].append(current_lr_G)
            train_hist['learning_rates_D'].append(current_lr_D)
            
            writer.add_scalars("Losses", {
                "Generator": np.mean(G_losses),
                "Discriminator": np.mean(D_losses),
                "Validation": val_loss,
                "L1": np.mean(l1_losses)
            }, epoch)
            
            writer.add_scalars("LearningRates", {
                "Generator": current_lr_G,
                "Discriminator": current_lr_D
            }, epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_dir = f'results/{experiment_name}/fold_{fold+1}_best'
                os.makedirs(save_dir, exist_ok=True)
                torch.save(generator.state_dict(), os.path.join(save_dir, 'generator.pth'))
                torch.save(discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))
    
    writer.close()
    
    save_dir = f'results/{experiment_name}/fold_{fold+1}_final'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(save_dir, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))
    
    return train_hist

