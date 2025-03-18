import torch
from torch import nn

from utils import fetch_model_dict

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()
        
    def forward(self, x):
        return x[:, :, :320, :320]

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=3, padding=1), # 107x107
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=1), # 36x36
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=1), # 12x12
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1), # 4x4, 4x4x128 = 2048
            nn.Flatten(),
        )    
        
        self.z_mean = nn.Linear(2048, 3)
        self.z_log_var = nn.Linear(2048, 3)
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 2048),
            Reshape(-1, 128, 4, 4),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3, padding=1, output_padding=2),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=3, padding=1, output_padding=2),                
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=3, padding=1, output_padding=1),                
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(16, 5, kernel_size=3, stride=3, padding=1, output_padding=1), 
            Trim(),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded

class ConvAutoencoder(nn.Module):
    def __init__(self):
        # N, 1, 28, 28
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7), # N, 64, 1, 1
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7), # N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class ConvAutoencoderLinear(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [batch, 32, 160, 160]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [batch, 64, 80, 80]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # [batch, 128, 40, 40]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # [batch, 256, 20, 20]
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 20 * 20, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, latent_dim)  # Latent vector
        )

        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256 * 20 * 20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (256, 20, 20))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # [batch, 128, 40, 40]
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # [batch, 64, 80, 80]
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 32, 160, 160]
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [batch, 1, 320, 320]
            nn.Sigmoid()  # Output range [0,1] for image reconstruction
        )

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.fc_enc(encoded)
        decoded = self.fc_dec(latent)
        output = self.decoder(decoded)
        return output

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(100000, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 100000),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_VAE():
    pass
    

if __name__ == "__main__":
    vae = VAE()
