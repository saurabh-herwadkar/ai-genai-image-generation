# PyTorch imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
 
# Data handling and visualization
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
 
# Utilities
from tqdm.notebook import tqdm
 
# Set up the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the transform for MNIST dataset
transform = transforms.Compose([
    transforms.Resize(32),  # Resize images to 32x32
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

 
batch_size = 32 
# Load MNIST dataset
dataset = MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def sinusoidal_embedding(n, d):
    """
    Creates sinusoidal embeddings for the time step.
 
    Args:
    n (int): Number of time steps
    d (int): Dimension of the embedding
 
    Returns:
    torch.Tensor: Sinusoidal embedding of shape (n, d)
    """
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])
 
    return embedding


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
 
    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_channels)
        if up:
            self.conv = nn.Conv2d(2*in_channels, out_channels, 3, 1, 1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        self.conv1 = ResidualConvBlock(out_channels, out_channels, is_res=True)
        self.conv2 = ResidualConvBlock(out_channels, out_channels, is_res=True)
        self.relu = nn.ReLU()
 
    def forward(self, x, t, ):
        # First Conv
        h = self.conv(x)
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.conv1(h)
        # Third Conv
        h = self.conv2(h)
        # Down or Upsample
        return self.transform(h)
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
 
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
 
        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, 64, 3, 1, 1)
 
        # Downsample
        self.downs = nn.ModuleList([
            UNetBlock(64, 128, time_dim),
            UNetBlock(128, 256, time_dim),
            UNetBlock(256, 512, time_dim),
            UNetBlock(512, 1024, time_dim),
        ])
 
        # Upsample
        self.ups = nn.ModuleList([
            UNetBlock(1024, 512, time_dim, up=True),
            UNetBlock(512, 256, time_dim, up=True),
            UNetBlock(256, 128, time_dim, up=True),
            UNetBlock(128, 64, time_dim, up=True),
        ])
 
        # Final conv
        self.output = nn.Conv2d(64, out_channels, 1)
 
    def forward(self, x, t):
        # Time embedding
        t = self.time_mlp(sinusoidal_embedding(t.shape[0], self.time_dim).to(x.device))
 
        # Initial conv
        x = self.conv0(x)
 
        # U-Net
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
 
        return self.output(x)
    

class DDPM(nn.Module):
    def __init__(self, network, num_timesteps=1000, beta_start=0.0001, 
                 beta_end=0.02, device="cuda"):
        super().__init__()
        self.network = network.to(device)
        self.num_timesteps = num_timesteps
        self.device = device
 
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
 
        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
 

