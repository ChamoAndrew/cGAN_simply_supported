import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Automatically choose GPU if available

# ================================
# 1️⃣ Latent Diffusion Noise Generator
# ================================
class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

        self.resize = transforms.Resize((64, 64))  # Resize images to 64x64

    def forward_process(self, x0, t):
        """Adds noise to images and outputs (batch_size, 256) latent vectors."""
        x0_resized = self.resize(x0)  # Convert (batch, 1, 512, 512) → (batch, 1, 64, 64)
        noise = torch.randn_like(x0_resized).to(device)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1).to(device)
        xt = (torch.sqrt(alpha_bar_t) * x0_resized + torch.sqrt(1 - alpha_bar_t) * noise)
        return xt  


# ================================
# 2️⃣ Generator (Denoising Network)
# ================================
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(conv_block, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(out_channels)
        )  
    def forward(self, x):
        return self.conv_block(x)

class transconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(transconv_block, self).__init__()
        self.transconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return self.transconv_block(x)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # First block for x
        self.for_x = nn.Sequential(
            transconv_block(1, 5),
            transconv_block(5, 10),
            transconv_block(10, 3)
        )
        
        # Encoder to process the image input (x)
        self.encoder = nn.Sequential(
            conv_block(1, 4),
            conv_block(4, 8),
            conv_block(8, 16),
            conv_block(16, 32),
            conv_block(32, 64, kernel_size=4, stride=1, padding=0)  # Output size: (1, 1, 64)
        )

        # Decoder to generate the final output image
        self.decoder = nn.Sequential(
            transconv_block(67, 64, kernel_size=4, stride=1, padding=0),
            transconv_block(64, 32, kernel_size=8, stride=4, padding=2),
            transconv_block(32, 16),
            transconv_block(16, 8, kernel_size=8, stride=4, padding=2),
            transconv_block(8, 8),
            transconv_block(8, 4)
        )

        # Last convolutional transpose to output the final image
        self.last = nn.ConvTranspose2d(7, 1, kernel_size=3, stride=1, padding=1)

        self.apply(self.weights_init)  # Apply weight initialization

    def forward(self, x, z):  # x -> image input, z -> labels
        x = x.view(-1, 1, 64, 64).float()
        z = z.view(-1, 3, 1, 1).float()  # Reshape labels to match required input shape for z
        latent = self.encoder(x)  # Output shape: (batch_size, 64, 1, 1)
        Z = torch.cat([z, latent], 1)  # Concatenate along the channel dimension, resulting in shape: (batch_size, 67, 1, 1)
        feed_x = self.for_x(x)
        before_out = torch.cat([feed_x, self.decoder(Z)], 1)
        out = self.last(before_out)  # Output image with shape: (batch_size, 1, 64, 64)
        return out
    
    def weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ================================
# 3️⃣ Discriminator (Classifies Real vs Fake Images)
# ================================
class Discriminator(nn.Module):
    def __init__(self, img_channels, label_dim):
        super(Discriminator, self).__init__()
        
        self.img_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1)),  # (256x256)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),  # (128x128)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),  # (64x64)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),  # (32x32)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(512, 1024, 4, 2, 1)),  # (16x16)
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(1024, 2048, 4, 2, 1)),  # (8x8)
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc_img = nn.Linear(2048 * 8 * 8, 1024)
        self.fc_label = nn.Linear(label_dim, 1024)
        self.fc_final = nn.utils.spectral_norm(nn.Linear(2048, 1))  # Spectral normalization for stability

        self.apply(self.weights_init)  # Apply weight initialization

    def forward(self, img, labels):
        img = img.to(device)  # Move img to the correct device
        labels = labels.to(device)  # Move labels to the correct device
        
        img_features = self.img_conv(img).view(img.size(0), -1)  # Flatten
        img_features = F.leaky_relu(self.fc_img(img_features), 0.2)
        label_features = F.leaky_relu(self.fc_label(labels), 0.2)
        combined = torch.cat([img_features, label_features], dim=1)
        validity = self.fc_final(combined)
        return validity

    def weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
