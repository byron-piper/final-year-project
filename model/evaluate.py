import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader as DL

from datasets import FlowField

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
    
    dataset = FlowField(datasets_folder, train=True)
    dataloader = DL(dataset=dataset, batch_size=5, shuffle=True, num_workers=0)
    
    aoa = np.linspace(-5, 30, 15)
    
    latent_vectors = None
    with torch.no_grad():
        for batch_idx, (flow, _) in enumerate(dataloader):    
            flow = flow.to(device)
            output = model(flow)
            latents = model.compute_latents(flow)
            
            flow = flow.detach().cpu().x.numpy()[:, 1]
            output = output.detach().cpu().numpy()[:, 1]
            latents = latents.detach().cpu().numpy()
            
            if latent_vectors is None: latent_vectors = latents
            else: latent_vectors = np.concat((latent_vectors, latents))
            
            print(batch_idx+1)
            
            if (batch_idx+1) % 3 == 0:
                plt.scatter(aoa, latent_vectors[:, 0])
                plt.show()
                latent_vectors = None
            
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')  # Standalone 3D plot

    # # Scatter plot
    # ax.scatter(latent_vectors[:, 0], latent_vectors[:, 1], latent_vectors[:, 2], c='blue', marker='o', alpha=0.6)

    # # Labels
    # ax.set_xlabel('X Axis')
    # ax.set_ylabel('Y Axis')
    # ax.set_zlabel('Z Axis')
    # ax.set_title('Standalone 3D Scatter Plot')

    # # Show plot
    # plt.show()
        
    return
    
def visualise_flowfield_reconstruction(params:dict, model_dict:dict):
    #region # ==== UNPACK PARAMETERS ==== #
    
    field_dim = params["preprocess"]["field_dim"]
    
    datasets_folder = params["i/o"]["datasets_folder"]
    
    batch_size = params["training"]["batch_size"] 
    
    #endregion
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_dict["model"].to(device)
    
    model.eval()
    
    aoa = np.linspace(-5, 30, 15)
    
    dataset = FlowField(datasets_folder, train=False)
    dataloader = DL(dataset=dataset, batch_size=5)
    lats = None
    with torch.no_grad():
        for batch_idx, (flow, xy) in enumerate(dataloader):
            flow = flow.to(device)
            #output = model(flow)
            
            #flow = flow.detach().cpu().x.numpy().reshape(5, 102400, 3)
            #output = output.detach().cpu().numpy().reshape(5, 102400, 3)
            #xy = xy.detach().cpu().numpy().reshape(5, 102400, 2)
            
            latents = model.compute_latents(flow)
            latents = latents.detach().cpu().numpy()
            
            if lats is None: lats = latents
            else: lats = np.concat((lats, latents))
            
            if (batch_idx+1) % 3 == 0:
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(projection='3d')
                # axes[0].scatter(aoa, lats[:, 0])
                # axes[0].grid()
                # axes[1].scatter(aoa, lats[:, 1])
                # axes[1].grid()
                # axes[2].scatter(aoa, lats[:, 2])
                # axes[2].grid()
                
                ax.scatter(lats[:, 0], lats[:, 1], lats[:, 2])
                
                plt.tight_layout()
                plt.show()
                lats = None
            
            # for i in range(latents.shape[0]):
            #     pass
            #     # x = xy[i][:, 0]
            #     # y = xy[i][:, 1]
            
            #     # triang = Triangulation(x, y)
                
            #     # axes[i, 0].tricontourf(triang, flow[i][:, 1], levels=100, cmap="jet")
            #     # axes[i, 0].axis("off")
            #     # axes[i, 0].set_title(f"Original Flowfield [{i+1}]")
                
            #     # axes[i, 1].tricontourf(triang, flow[i][:, 1], levels=100, cmap="jet")
            #     # axes[i, 1].axis("off")
            #     # axes[i, 1].set_title(f"Reconstructed Flowfield [{i+1}]")
            #     axes[0].scatter(latents[i][0], latents[i][1])
            #     axes[1].scatter(latents[i][0], latents[i][2])
            #     axes[2].scatter(latents[i][1], latents[i][2])
            # plt.tight_layout()
            # plt.show()
                
if __name__ == "__main__":
    params = load_parameters()
    
    checkpoint = params["model"]["checkpoint"]
    
    model_dict = fetch_model_dict(params, checkpoint)
    
    visualise_flowfield_reconstruction(params, model_dict)