import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 64
 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
 
train_dataset = datasets.MNIST(root='./data', train=True, download=True, 
                               transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
 
        self.model = nn.Sequential(
            # Fully connected layer that reshapes the input noise vector
            nn.Linear(100, 256 * 7 * 7),
            nn.Unflatten(1, (256, 7, 7)),
 
            # Transposed Convolutional Layer
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, 
                               padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
 
            # Another Transposed Convolutional Layer
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
 
            # Final Transposed Convolutional Layer
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, 
                               padding=1, output_padding=1),
            nn.Tanh()
        )
 
    def forward(self, z):
        return self.model(z)
    

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
 
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
 
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0),
            nn.Flatten()
        )
 
    def forward(self, x):
        return self.model(x)
    
# Initialize models
critic = Critic().to(device)
generator = Generator().to(device)
 
# Optimizers
critic_optimizer = optim.Adam(critic.parameters(), 
                              lr=0.0001, betas=(0.6, 0.999))
generator_optimizer = optim.Adam(generator.parameters(), 
                                 lr=0.0001, betas=(0.6, 0.999))

# Hyperparameters
num_epochs = 250
clip_value = 1 # clipping value for the critic's weight
n_critic = 2   # number of critic updates per generator update
 

def run_wgan():

    # Training Loop
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(train_loader):
    
            # ---------------------
            #  Train Critic
            # ---------------------
            
            # Load real images
            real_images = images.to(device)   
    
            critic_optimizer.zero_grad()
    
            # Generate a batch of images
            z = torch.randn(images.size(0), 100).to(device)  # Generate noise vector
            fake_images = generator(z)
    
            # Calculate critic's loss on real and fake images
            critic_real = critic(real_images).reshape(-1)
            critic_fake = critic(fake_images).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            loss_critic.backward()
            critic_optimizer.step()
    
            # Clip weights of critic
            for p in critic.parameters():
                p.data.clamp_(-clip_value, clip_value)
    
            # Train the generator every n_critic iterations
            if i % n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                generator_optimizer.zero_grad()
    
                # Generate a batch of images
                fake_images = generator(z)
    
                # Calculate loss on generator's images
                critic_fake = critic(fake_images).reshape(-1)
                loss_generator = -torch.mean(critic_fake)
                loss_generator.backward()
                generator_optimizer.step()
    
            # Print out the losses occasionally
            if i % 2000 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(train_loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_generator:.4f}")