import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import VAE
from datasets import FluentDataset

def BCE_loss_fn(x_hat, x, mu, logvar):
    
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x, reduction='sum'
    )

    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

    return BCE + KLD

def train():
    with open("parameters.json", "r") as f:
        parameters = json.load(f)
    
    #region # ==== UNPACK PARAMETERS ==== #
    
    # Training
    device_tag = parameters["training"]["device_tag"]
    batch_size = parameters["training"]["batch_size"]    
    learning_rate = parameters["training"]["learning_rate"]
    epochs = parameters["training"]["epochs"]
    loss_fn = parameters["training"]["loss_fn"]
    
    # Model parameters
    conv_hid_dim = parameters["training"]["conv_hid_dim"]
    hid_lat_dim = parameters["training"]["hid_lat_dim"]
    
    # I/O
    datasets_folder = parameters["i/o"]["datasets_folder"]
    
    #endregion
    
    dataset = FluentDataset(datasets_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device(device_tag)
    
    model = VAE(conv_hid_dim, hid_lat_dim).to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in dataloader:
            x = batch[0]
            x = x.to(device)
            optimiser.zero_grad()
            
            x_recon, mu, sigma = model(x)
            
            if loss_fn == "BCE":
                loss = BCE_loss_fn(x_recon, x, mu, sigma)
            
            loss.backward()
            optimiser.step()
            
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

if __name__ == "__main__":
    train()