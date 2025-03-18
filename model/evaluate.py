import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import FlowFieldDataset

sys.path.append(r"C:\Users\honey\Documents\PROJECT\final-year-project")

from sample.helper import load_parameters
from utils import fetch_model_dict

def visualise_model(params:dict, model_dict:dict):
    #region # ==== UNPACK PARAMETERS ==== #
    
    field_dim = params["preprocess"]["field_dim"]
    
    datasets_folder = params["i/o"]["datasets_folder"]
    
    batch_size = params["training"]["batch_size"] 
    
    #endregion
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_dict["model"].to(device)
    
    model.eval()
    
    dataset = FlowFieldDataset(datasets_folder, train=True)
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=5)

    latents = []
    with torch.no_grad():
        for x, _, _ in dataloader:
            x = x.view(batch_size, 1, field_dim, field_dim)
            x = x.to(device)
            latent = model.fc_enc(model.encoder(x))
            latents.append(latent.cpu().numpy())
    
    latents = np.concat(latents, axis=0)
    
    latents = np.array(latents)
    
    x = latents[:, 0]
    y = latents[:, 1]
    z = latents[:, 2]

    # Create 3D scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, alpha=0.5, c=np.arange(len(x)), cmap='viridis')

    # Labels and title
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')
    ax.set_title('3D Latent Space Visualization')

    plt.show()

def visualise_vae(params:dict, model_dict:dict):
    #region # ==== UNPACK PARAMETERS ==== #
    
    field_dim = params["preprocess"]["field_dim"]
    
    datasets_folder = params["i/o"]["datasets_folder"]
    
    batch_size = params["training"]["batch_size"] 
    
    #endregion
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_dict["model"].to(device)
    
    model.eval()
    
    dataset = FlowFieldDataset(datasets_folder, train=True)
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=5)
    
    fig, axes = plt.subplots(nrows=2, ncols=1, 
                             sharex=True, sharey=True)
    
    for batch_idx, (features, _, _) in enumerate(dataloader):
        
        features = features.to(device)

        color_channels = features.shape[1]
        image_height = features.shape[2]
        image_width = features.shape[3]
        
        with torch.no_grad():
            encoded, z_mean, z_log_var, decoded_images = model(features)

        orig_images = features[:0]
        break
    
    for i in range(1):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img.detach().to(torch.device('cpu'))        

            ax.imshow(curr_img.view((image_height, image_width)), cmap='binary')

if __name__ == "__main__":
    params = load_parameters()
    
    checkpoint = params["model"]["checkpoint"]
    
    model_dict = fetch_model_dict(params, checkpoint)
    
    visualise_vae(params, model_dict)