import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax, scatter
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TopKPooling, global_mean_pool, global_max_pool
from torch_geometric.nn.models import GAE, InnerProductDecoder

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
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 80x80
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 40x40
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 20x20
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 10x10
            nn.ReLU()
        )

        self.fc_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25600, 2048), # 10x10x256
            nn.ReLU(),
            nn.Linear(2048, 512), # 10x10x256
            nn.ReLU(),
            nn.Linear(512, 128)  # Latent vector
        )

        self.fc_dec = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 25600),
            nn.ReLU(),
            nn.Unflatten(1, (256, 10, 10))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 20x20
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 40x40
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 80x80
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   # 160x160
            nn.ReLU(),
            nn.ConvTranspose2d(16, 5, kernel_size=3, stride=2, padding=1, output_padding=1),   # 320x320
            nn.Sigmoid()
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

class GlobalAttention(torch.nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """

    def __init__(self, gate_nn, nn=None):
        super(GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn


    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(src=gate, index=batch, num_nodes=size)
        out = scatter(reduce='add', src=gate * x, index=batch, dim_size=size)

        return out


    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)

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

class GCNLinear(nn.Module):
    def __init__(self, in_channels, hid_channels, emb_channels):
        super(GCNLinear, self).__init__()

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.emb_channels = emb_channels
        
        self.encoder = nn.Sequential(
            GCNConv(in_channels=in_channels, out_channels=hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(),
            GCNConv(in_channels=hid_channels, out_channels=hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(),
            GCNConv(in_channels=hid_channels, out_channels=hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(),
            GCNConv(in_channels=hid_channels, out_channels=hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU()
        )
        
        self.fc_latent = nn.Sequential(
            nn.Linear(hid_channels, hid_channels//2),
            nn.ReLU(),
            nn.Linear(hid_channels//2, emb_channels)
        )
    
        self.fc_decoder = nn.Sequential(
            nn.Linear(emb_channels, hid_channels//2),
            nn.ReLU(),
            nn.Linear(hid_channels//2, hid_channels),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            GCNConv(in_channels=hid_channels, out_channels=hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(),
            GCNConv(in_channels=hid_channels, out_channels=hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(),
            GCNConv(in_channels=hid_channels, out_channels=hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(),
            GCNConv(in_channels=hid_channels, out_channels=in_channels),
            nn.Sigmoid()
        )
        
    def get_channels(self):
        return self.in_channels, self.hid_channels, self.emb_channels
        
    def encode(self, x, edge_index, batch_index):
        # Apply convolutional layers, 
        # N : [batch_size * num_features, 5]  ->
        # N:  [64, 64]
        for layer in self.encoder:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)

        # Apply global mean pooling to get vector representation
        # N : [64, 64] ->
        # N : [batch_size, 64]
        x = global_mean_pool(x, batch_index)
        
        # Apply dense layers
        # N : [batch_size, 64] ->
        # N : [batch_size, 3]
        x = self.fc_latent(x)
        
        return x
        
    def decode(self, x, edge_index, batch_index):
        #x = x[batch_index]
        
        # Dense layers -> x.shape = [5, 64]
        x = self.fc_decoder(x)
        
        # Inverse pooling? -> x.shape = [500000x64]
        x = x[batch_index]
         
        # decode
        for layer in self.decoder:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)

        return x

    def forward(self, data):
        x, edge_index, batch_index = data.x, data.edge_index, data.batch
        
        embedding = self.encode(x, edge_index, batch_index)
        x = self.decode(embedding, edge_index, batch_index)
                
        return x, embedding 

class VGAE(nn.Module):
    def __init__(self):
        super(VGAE, self).__init__()
        
        self.encoder = nn.Sequential(
            GCNConv(in_channels=5, out_channels=32),
            nn.ReLU(),
            GCNConv(in_channels=32, out_channels=32),
            nn.ReLU(),
            GCNConv(in_channels=32, out_channels=32),
            nn.ReLU(),
            GCNConv(in_channels=32, out_channels=32),
            nn.ReLU()
        )
        
        self.embed_mu = nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=3)
        )
        
        self.embed_sigma = nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=3)
        )
        
        self.unembed = nn.Sequential(
            nn.Linear(in_features=3, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            GCNConv(in_channels=32, out_channels=32),
            nn.ReLU(),
            GCNConv(in_channels=32, out_channels=32),
            nn.ReLU(),
            GCNConv(in_channels=32, out_channels=32),
            nn.ReLU(),
            GCNConv(in_channels=32, out_channels=5),
            nn.ReLU()
        )
        
        self.pooling = global_mean_pool
        
    def encode(self, x, edge_index, batch_index):
        for layer in self.encoder:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        
        mu = self.embed_mu(x)
        sigma = self.embed_sigma(x)
        
        mu = self.pooling(mu, batch_index)
        sigma = self.pooling(sigma, batch_index)
        
        return mu, sigma
    
    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, edge_index, batch_index):
        z = z[batch_index]
        
        z = self.unembed(z)
        for layer in self.decoder:
            if isinstance(layer, GCNConv):
                z = layer(z, edge_index)
            else:
                z = layer(z)
        
        return z
    
    def forward(self, data):
        x, edge_index, batch_index = data.x, data.edge_index, data.batch
        
        mu, sigma = self.encode(x, edge_index, batch_index)
        z = self.reparameterize(mu, sigma)
        x_recon = self.decode(z, edge_index, batch_index)
        return x_recon, mu, sigma

class GraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, k=0.5):
        super(GraphAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.conv1 = SAGEConv(self.in_channels, self.hidden_dim)
        #self.pool1 = TopKPooling(self.hidden_dim, ratio=k)
        
        self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim)
        #self.pool2 = TopKPooling(self.hidden_dim, ratio=k)
        
        self.fc_latent = nn.Linear(self.hidden_dim*2, self.latent_dim)  # Latent representation
        
        # Decoder
        self.fc_decode = nn.Linear(self.latent_dim, self.hidden_dim*2)
        self.deconv1 = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.deconv2 = SAGEConv(self.hidden_dim, self.in_channels)
        
    def forward(self, x, edge_index, batch):
        # Encoder
        x = F.relu(self.conv1(x, edge_index))
        print(x.shape)
        #x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        #pooled1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        
        x = F.relu(self.conv2(x, edge_index))
        print(x.shape)
        #x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        print(pooled.shape)
        # Skip connection
        embedding = self.fc_latent(pooled)
        print(embedding.shape)
        
        # Decoder
        x = F.relu(self.fc_decode(embedding))
        print(x.shape)
        x = x.view(-1, self.hidden_dim)
        print(x.shape)
        x = F.relu(self.deconv1(x, edge_index))
        print(x.shape)
        x = self.deconv2(x, edge_index)
        print(x.shape)
        
        return x, embedding