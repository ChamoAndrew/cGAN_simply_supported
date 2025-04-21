import cv2
import numpy as np
import torch
import torch.nn as nn
from src.models import Generator, Discriminator, GaussianDiffusion
from sklearn.model_selection import train_test_split
from src.utils import dataset_creation
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import os


def test_generator(csv_path: str, target_folder: str, batch_size=19, 
                   img_channels=1, label_dim=3) -> None:
    Data_tg = dataset_creation(csv_path, target_folder)
    _, test_indices = train_test_split(range(len(Data_tg)), test_size=0.2, random_state=42)
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(Data_tg, batch_size=batch_size, sampler=test_sampler, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate models
    generator = Generator().to(device)
    discriminator = Discriminator(img_channels, label_dim).to(device)
    diffusion = GaussianDiffusion().to(device)
    
    generator.load_state_dict(torch.load('results\\fold_4\\generator.pth'))  # Load trained generator
    discriminator.load_state_dict(torch.load('results\\fold_4\\discriminator.pth')) # Load trained discriminator
    generator.eval()
    discriminator.eval()

    save_folder = "results/test_generated_binary"
    os.makedirs(save_folder, exist_ok=True)

    # Loss function for real/fake
    criterion_real_fake = nn.BCEWithLogitsLoss().to(device)

    test_loss_real_fake = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, (labels, imgs) in tqdm(enumerate(test_loader), desc="Testing Generator ..."):
            imgs = imgs.to(device)
            labels = labels.to(device)
            t = torch.randint(0, diffusion.timesteps, (imgs.size(0),), device=device)
            noise = diffusion.forward_process(imgs, t)
            gen_imgs = generator(noise, labels)

            # Get real/fake validity score
            validity_fake = discriminator(gen_imgs, labels)

            # Compute real/fake loss
            loss_real_fake = criterion_real_fake(validity_fake, torch.ones_like(validity_fake)) 

            # Accumulate losses
            test_loss_real_fake += loss_real_fake.item() * labels.size(0)
            total_samples += labels.size(0)
            
            # Convert images to binary and save
            gen_imgs = gen_imgs.cpu().numpy()
            for j, img in enumerate(gen_imgs):
                # Normalize to [0, 1] then threshold to binary
                img = (img[0] + 1) / 2  # Convert from [-1,1] to [0,1]
                img_binary = (img > 0.5).astype(np.uint8) * 255  # Threshold and scale to 0 or 255
                
                # Save as binary PNG (lossless)
                cv2.imwrite(
                    os.path.join(save_folder, f"test_img_{i * batch_size + j}.png"), 
                    img_binary,
                    [cv2.IMWRITE_PNG_BILEVEL, 1]  # Force binary PNG
                )

    # Compute final losses
    final_test_loss_real_fake = test_loss_real_fake / total_samples
    print(f"\n✅ Final Test Loss (Real/Fake): {final_test_loss_real_fake:.4f}")
    print(f"\nBinary images saved in '{save_folder}'")


if __name__ == "__main__":
    test_generator("Simply_supported\\Data.csv", "Simply_supported\\Target")