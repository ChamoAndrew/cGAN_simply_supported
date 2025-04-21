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
    
    generator.load_state_dict(torch.load('results/fold_4/generator.pth'))  # Load trained generator
    discriminator.load_state_dict(torch.load('results/fold_4/discriminator.pth')) # Load trained discriminator
    generator.eval()
    discriminator.eval()

    # Create required directories
    save_dir = "results/test_generated"
    os.makedirs(f"{save_dir}/images", exist_ok=True)
    os.makedirs(f"{save_dir}/ground_truth", exist_ok=True)
    os.makedirs(f"{save_dir}/losses", exist_ok=True)
    os.makedirs(f"{save_dir}/labels", exist_ok=True)

    # Loss function for binary image comparison
    criterion_binary = nn.BCELoss().to(device)

    selected_images = []  # Store samples for later saving
    selected_labels = []
    selected_losses = []

    with torch.no_grad():
        all_images = []
        all_labels = []
        all_losses = []

        for i, (labels, imgs) in tqdm(enumerate(test_loader), desc="Testing Generator ..."):
            imgs = imgs.to(device)
            labels = labels.to(device)
            t = torch.randint(0, diffusion.timesteps, (imgs.size(0),), device=device)
            noise = diffusion.forward_process(imgs, t)
            gen_imgs = generator(noise, labels)

            # Convert to binary format (thresholding)
            gen_imgs_bin = torch.where(gen_imgs > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
            real_imgs_bin = torch.where(imgs > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

            # Compute binary loss (BCE on binary images)
            loss_binary = criterion_binary(gen_imgs_bin, real_imgs_bin)

            # Store all samples
            all_images.extend(list(zip(gen_imgs_bin.cpu().numpy(), real_imgs_bin.cpu().numpy())))
            all_labels.extend(labels.cpu().numpy())
            all_losses.extend([loss_binary.item()] * labels.size(0))

        # Select 4 evenly spaced images across the dataset
        num_samples = len(all_images)
        selected_indices = np.linspace(0, num_samples - 1, num=4, dtype=int)
        
        for idx in selected_indices:
            selected_images.append(all_images[idx])
            selected_labels.append(all_labels[idx])
            selected_losses.append(all_losses[idx])

        # Save selected images
        for j, ((gen_img, real_img), label, loss) in enumerate(zip(selected_images, selected_labels, selected_losses)):
            img_gen = (gen_img[0] * 255).astype(np.uint8)  # Convert to binary image (0 or 255)
            img_real = (real_img[0] * 255).astype(np.uint8)  # Convert to binary image (0 or 255)

            # Save generated and ground truth images
            cv2.imwrite(f"{save_dir}/images/gen_img_{j}.png", img_gen)
            cv2.imwrite(f"{save_dir}/ground_truth/real_img_{j}.png", img_real)

            # Save loss
            with open(f"{save_dir}/losses/loss_{j}.txt", "w") as f:
                f.write(f"Binary BCE Loss: {loss:.4f}")

            # Save labels
            with open(f"{save_dir}/labels/label_{j}.txt", "w") as f:
                force, length, height = label  # Extract normalized values
                f.write(f"Force: {force * 200} kN\n")
                f.write(f"Length: {length * 6} m\n")
                f.write(f"Height: {height * 3} m\n")

    print(f"\n✅ 4 Binary Images Saved in '{save_dir}/'")    

if __name__ == "__main__":
    test_generator("Simply_supported/Data.csv", "Simply_supported/Target")
