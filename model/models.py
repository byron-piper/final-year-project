import torch
from torch import nn
from torch_geometric import nn as nng

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
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies=4):
        """
        Positional encoding for CFD mesh node (x, y) values.
        :param num_frequencies: Number of frequency components for sine/cosine encoding.
        """
        super().__init__()
        self.num_frequencies = num_frequencies  # How many frequency components to use

    def forward(self, x, y):
        """
        :param x: Tensor of x-coordinates, shape (B, H, W, 1)
        :param y: Tensor of y-coordinates, shape (B, H, W, 1)
        :return: Encoded positional tensor, shape (B, H, W, 2*num_frequencies)
        """
        freq_bands = 2.0 ** torch.linspace(0.0, self.num_frequencies - 1, self.num_frequencies)  # Log-spaced frequencies

        # Apply sin and cos encoding
        pos_enc_x = torch.cat([torch.sin(x * f) for f in freq_bands] + 
                              [torch.cos(x * f) for f in freq_bands], dim=-1)
        pos_enc_y = torch.cat([torch.sin(y * f) for f in freq_bands] + 
                              [torch.cos(y * f) for f in freq_bands], dim=-1)

        # Combine x and y encodings
        pos_encoding = torch.cat([pos_enc_x, pos_enc_y], dim=-1)  # Shape (B, H, W, 2 * num_frequencies * 2)
        return pos_encoding

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
            nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1),  # 160x160
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=1), # 54x54
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1), # 18x18
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=3, padding=1), # 6x6
            nn.LeakyReLU(0.2, inplace=True), 
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=3, padding=1, output_padding=2), # 18x18
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3, padding=1, output_padding=2), # 54x54
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=3, padding=1, output_padding=0),  # 160x160
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 5, kernel_size=3, stride=2, padding=1, output_padding=1),   # 320x320
            Trim(),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class ConvAutoencoderLinear(nn.Module):
    def __init__(self):
        super().__init__()    
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=2, padding=1),  # 160x160
            nn.ELU(alpha=1.0, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 80x80
            nn.ELU(alpha=1.0, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 40x40
            nn.ELU(alpha=1.0, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 20x20
            nn.ELU(alpha=1.0, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 10x10
            nn.ELU(alpha=1.0, inplace=True),
        )

        self.fc_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25600, 2048), # 10x10x256
            nn.ELU(alpha=1.0, inplace=True),
            nn.Linear(2048, 512), # 10x10x256
            nn.ELU(alpha=1.0, inplace=True),
            nn.Linear(512, 128)  # Latent vector
        )

        self.fc_dec = nn.Sequential(
            nn.Linear(128, 512),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Linear(512, 2048),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Linear(2048, 25600),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Unflatten(1, (256, 10, 10))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 20x20
            nn.ELU(alpha=1.0, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 40x40
            nn.ELU(alpha=1.0, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 80x80
            nn.ELU(alpha=1.0, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   # 160x160
            nn.ELU(alpha=1.0, inplace=True),
            nn.ConvTranspose2d(16, 5, kernel_size=3, stride=2, padding=1, output_padding=1),   # 320x320
            Trim(),
            nn.Tanh()
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

class GCCN(nn.Module):
    def __init__(self, in_channels):
        super(GCCN, self).__init__()

        self.encoder = nn.Sequential(
            nng.GCNConv(in_channels=in_channels, out_channels=160),
            nn.ReLU(),
            nng.GCNConv(in_channels=160, out_channels=80),
            nn.ReLU(),
            nng.GCNConv(in_channels=80, out_channels=40),
            nn.ReLU(),
            nng.GCNConv(in_channels=40, out_channels=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=3)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=3, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=20),
            nn.ReLU(),
            nng.GCNConv(in_channels=20, out_channels=40),
            nn.ReLU(),
            nng.GCNConv(in_channels=40, out_channels=80),
            nn.ReLU(),
            nng.GCNConv(in_channels=80, out_channels=160),
            nn.ReLU(),
            nng.GCNConv(in_channels=160, out_channels=3),
            nn.Tanh()
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        
        # Encode
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nng.GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        
        # Decode
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nng.GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
                
        return x

class GCCNLinear(nn.Module):
    def __init__(self, in_channels):
        super(GCCNLinear, self).__init__()

        self.global_pool = nng.global_mean_pool

        self.encoder = nn.Sequential(
            nng.GCNConv(in_channels=in_channels, out_channels=64),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.2),
            nng.GCNConv(in_channels=64, out_channels=128),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.2),
            nng.GCNConv(in_channels=128, out_channels=64),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.2)
        )
        
        self.fc_latent = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.2),
            nn.Linear(32, 3)
        )
    
        self.fc_decoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.2)
        )
        
        self.decoder = nn.Sequential(
            nng.GCNConv(64, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.2),
            nng.GCNConv(128, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.2),
            nng.GCNConv(64, in_channels), 
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.2),
            nn.Tanh()
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        batch = torch.arange(data.x.size(0), device=data.x.device)

        # Encode
        for layer in self.encoder:
            if isinstance(layer, nng.GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
                
        x = self.fc_latent(x)
        x = nng.global_max_pool(x, batch)
        x = self.fc_decoder(x)
        
        # Decode
        for layer in self.decoder:
            if isinstance(layer, nng.GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
                
        return x
    
    def compute_latents(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        
        # Encode
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nng.GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
                
        x = self.fc_latent(x)
        
        x = nng.global_mean_pool(x, batch)
                
        return x   
