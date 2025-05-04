import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
 

# Define transformations for the data
transform = transforms.Compose([
    transforms.ToTensor(),
])
 
# Load the MNIST dataset with labels
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
 
# Create DataLoader
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class cVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10, label_embedding_dim=50):
        super(cVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_embedding_dim = label_embedding_dim
        self.img_channels = 1  # Grayscale images
 
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, label_embedding_dim)
 
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.img_channels + label_embedding_dim, 32, kernel_size=4, stride=2, padding=1),  # Output: 32 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 64 x 7 x 7
            nn.ReLU(),
            nn.Flatten()
        )
 
        # Compute mu and logvar
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
 
        # Decoder input layer
        self.decoder_input = nn.Linear(latent_dim + label_embedding_dim, 64 * 7 * 7)
 
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32 x 14 x 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.img_channels, kernel_size=4, stride=2, padding=1),  # Output: 1 x 28 x 28
            nn.Sigmoid()
        )