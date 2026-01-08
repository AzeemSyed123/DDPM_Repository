# -*- coding: utf-8 -*-
"""DDPM_Image_Generation.py

Importing necessary libraries like `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`, `torchvision.datasets`, `torchvision.transforms`, `torchvision.utils`, `matplotlib.pyplot`, `tqdm`, `math`, and `os`. It also checks for GPU availability and prints the GPU name if available.
"""

import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import math
import os

"""Defining the `DDPMScheduler` class, which handles the noise scheduling for the Diffusion Probabilistic Model (DDPM). It sets up linear beta schedules, precomputes values, and provides methods to `add_noise` (forward diffusion) and `sample_prev_timestep` (reverse diffusion)."""

# ============================================================================
# DDPM Scheduler
# ============================================================================

class DDPMScheduler:
    """Handles noise scheduling for diffusion process"""

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device),
                                               self.alphas_cumprod[:-1]])

        # Precompute values for efficiency
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) /
                                  (1.0 - self.alphas_cumprod))

    def add_noise(self, x_start, timesteps, noise=None):
        """Forward diffusion: Add noise to images"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sample_prev_timestep(self, x_t, timesteps, predicted_noise):
        """Reverse diffusion: Remove noise from images"""
        betas_t = self.betas[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[timesteps]).view(-1, 1, 1, 1)

        # Predict x_0
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if timesteps[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[timesteps].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

"""Defining two core components of the U-Net model: `SinusoidalPositionEmbeddings` for time step embeddings and `ResidualBlock` for residual connections with time conditioning. It also includes an `AttentionBlock` for self-attention within the U-Net."""

# ============================================================================
# U-Net Model Components
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Time step embeddings"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time conditioning"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(min(32, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Multi-head attention
        q = q.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)

        scale = (c // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        out = self.proj(out)
        return x + out

"""Defining the `UNet` class, which is the main noise prediction model for the DDPM. It integrates the time embeddings, residual blocks, and attention blocks to create a U-shaped architecture for processing image data, including downsampling, a middle block with attention, and upsampling."""

class UNet(nn.Module):
    """U-Net for noise prediction"""
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_levels=(1, 2),
        dropout=0.1,
        time_emb_dim=512
    ):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Simplified architecture without skip connections
        ch = base_channels

        # Downsample
        self.down1 = ResidualBlock(ch, ch * 2, time_emb_dim, dropout)
        self.down2 = nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1)
        self.down3 = ResidualBlock(ch * 2, ch * 4, time_emb_dim, dropout)
        self.down4 = nn.Conv2d(ch * 4, ch * 4, 3, stride=2, padding=1)

        # Middle
        self.mid1 = ResidualBlock(ch * 4, ch * 4, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch * 4)
        self.mid2 = ResidualBlock(ch * 4, ch * 4, time_emb_dim, dropout)

        # Upsample
        self.up1 = nn.ConvTranspose2d(ch * 4, ch * 4, 4, stride=2, padding=1)
        self.up2 = ResidualBlock(ch * 4, ch * 2, time_emb_dim, dropout)
        self.up3 = nn.ConvTranspose2d(ch * 2, ch * 2, 4, stride=2, padding=1)
        self.up4 = ResidualBlock(ch * 2, ch, time_emb_dim, dropout)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

    def forward(self, x, timesteps):
        t_emb = self.time_mlp(timesteps)

        h = self.conv_in(x)

        # Downsample
        h = self.down1(h, t_emb)
        h = self.down2(h)
        h = self.down3(h, t_emb)
        h = self.down4(h)

        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Upsample
        h = self.up1(h)
        h = self.up2(h, t_emb)
        h = self.up3(h)
        h = self.up4(h, t_emb)

        return self.conv_out(h)

"""Defining helper functions for the training process. `train_epoch` performs a single training epoch, calculating loss and updating model parameters. `sample_images` generates new images from noise using the trained model and scheduler."""

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, scheduler, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()

    progress_bar = tqdm(dataloader, desc='Training')
    for images, _ in progress_bar:
        images = images.to(device)
        batch_size = images.shape[0]

        # Random timesteps
        timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device).long()

        # Add noise
        noise = torch.randn_like(images)
        noisy_images = scheduler.add_noise(images, timesteps, noise)

        # Predict noise
        predicted_noise = model(noisy_images, timesteps)

        # Loss
        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


@torch.no_grad()
def sample_images(model, scheduler, num_samples, device):
    """Generate images from noise"""
    model.eval()
    x = torch.randn(num_samples, 3, 32, 32).to(device)

    for t in tqdm(reversed(range(scheduler.num_timesteps)), desc='Sampling', total=scheduler.num_timesteps):
        timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, timesteps)
        x = scheduler.sample_prev_timestep(x, timesteps, predicted_noise)

    # Denormalize
    x = (x + 1) / 2
    return torch.clamp(x, 0, 1)


def show_images(images, title="Generated Images"):
    """Display images in notebook"""
    grid = make_grid(images, nrow=4, padding=2)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.title(title)
    plt.axis('off')
    plt.show()



"""Defining the `main` function, which orchestrates the entire DDPM training process. It sets up configuration parameters, loads and preprocesses the CIFAR-10 dataset, initializes the U-Net model, DDPM scheduler, and optimizer. The training loop runs for a specified number of epochs, generating samples periodically and saving checkpoints. Finally, it generates final samples, plots the training loss, and saves the trained model."""

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    num_epochs = 500
    lr = 2e-4

    print(f" Starting DDPM Training on {device}")
    print("="*60)

    # Load CIFAR-10
    print(" Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    print(f" Dataset loaded: {len(dataset)} images")

    # Initialize model
    print("\n Building U-Net model...")
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_levels=(1, 2),
        dropout=0.1
    ).to(device)

    scheduler = DDPMScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f" Model created: {num_params:,} parameters")

    # Training loop
    print(f"\n Training for {num_epochs} epochs...")
    print("="*60)

    losses = []
    for epoch in range(num_epochs):
        loss = train_epoch(model, scheduler, dataloader, optimizer, device)
        losses.append(loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.6f}")

        # Generate samples every 100 epochs
        if (epoch + 1) % 100 == 0 or (epoch + 1) == num_epochs:
            print(f"\n Generating samples at epoch {epoch+1}...")
            samples = sample_images(model, scheduler, 16, device)
            show_images(samples, f"Samples at Epoch {epoch+1}")

            # Save checkpoint locally (no Google Drive)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss
            }, f'checkpoint_epoch_{epoch+1}.pt')
            print(f" Checkpoint saved locally!\n")

    # Final samples
    print("\n Training Complete!")
    print("="*60)
    print("\n Generating final samples...")
    final_samples = sample_images(model, scheduler, 64, device)
    show_images(final_samples, "Final Generated Images (64 samples)")

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

    # Save final model for demo
    torch.save(model.state_dict(), 'final_model_500epochs.pt')
    print("\n Final model saved as 'final_model_500epochs.pt'")

    return model, scheduler, losses

"""This serves as the entry point for the script. It calls the `main` function to start the training process and retrieve the trained model, scheduler, and loss history. After training, it proceeds to generate and display additional samples using the trained model to demonstrate its capabilities."""

if __name__ == '__main__':
    model, scheduler, losses = main()
    print("\n Generating more images using the trained model.")

    # Generate more samples
    print("\n Generating additional samples...")
    new_samples = sample_images(model, scheduler, 16, 'cuda')
    show_images(new_samples, "Additional Generated Samples")
