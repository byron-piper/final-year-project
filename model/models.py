from torch import nn
from torch.functional import F

class VAE(nn.Module):
    def __init__(self, conv_hid_dim, hid_lat_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.conv_2hid = nn.Linear(conv_hid_dim[0], conv_hid_dim[1])
        
        self.hid_2mu = nn.Linear(hid_lat_dim[0], hid_lat_dim[1])
        self.hid_2sigma = nn.Linear(hid_lat_dim[0], hid_lat_dim[1])

        self.z_2hid1 = nn.Linear(hid_lat_dim[1], hid_lat_dim[0])
        self.hid_2conv = nn.Linear(conv_hid_dim[1], conv_hid_dim[0])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 5, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        #x = torch.flatten(x, start_dim=1)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.conv_2hid(x))
        
        mu = F.relu(self.hid_2mu(x))
        sigma = F.relu(self.hid_2sigma(x))
        
        return mu, sigma
    
    def reparametrise(self, mu, sigma):
        if self.training:
            std = sigma.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        x = F.relu(self.z_2hid(z))
        x = F.relu(self.hid_2conv(x))
        
        x = x.view(-1, 256, 256//16, 256//16)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparametrise(mu, sigma)
        x_recon = self.decode(z)

        return x_recon, mu, sigma
    
if __name__ == "__main__":
    vae = VAE()
