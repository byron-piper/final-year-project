from datetime import timedelta
import gzip
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.path import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Polygon, LineString

sys.path.append(r"C:\Users\honey\Documents\PROJECT\final-year-project")

from models import GCNLinear
from datasets import FlowField
from sample.helper import load_parameters
from utils import fetch_checkpoint

def plot_training_loss(checkpoint:dict):
    #region # ==== UNPACK CHECKPOINT ==== #
    
    model_id = checkpoint["model_id"]
    training_time = checkpoint["training_time"]
    
    loss_per_batch = checkpoint["loss_per_batch"]
    num_epochs = checkpoint["num_epochs"]
    num_epochs += 1
    
    num_batch_losses = len(loss_per_batch)
    iter_per_epoch = len(loss_per_batch) // num_epochs
    
    #endregion
    
    plt.figure()
    ax_loss = plt.subplot(1, 1, 1)
    # Set axis labels
    ax_loss.set_xlabel('Iterations')
    ax_loss.set_ylabel('Loss')
        
    loss_per_batch = checkpoint["loss_per_batch"]
    num_batch_losses = len(loss_per_batch)
    
    iter_per_epoch = len(loss_per_batch) // num_epochs
    
    # Create plot of total iterations (batch_size * num_epochs) against
    # loss for each batch
    ax_loss.plot(
        range(num_batch_losses), 
        loss_per_batch,
        label=f"{model_id} | Loss"
    )
    
    ax_loss.plot(
        np.convolve(loss_per_batch, 
                    np.ones(25,)/25, 
                    mode='valid'), 
        label=f"{model_id} | Running Average"
    )
        
    plt.legend()
        
    ax_avgloss = ax_loss.twiny()
    newlabel = list(range(num_epochs+1))
    
    newpos = [e*iter_per_epoch for e in newlabel]

    ax_avgloss.set_xticks(newpos[::num_epochs])
    ax_avgloss.set_xticklabels(newlabel[::num_epochs])

    ax_avgloss.xaxis.set_ticks_position('bottom')
    ax_avgloss.xaxis.set_label_position('bottom')
    ax_avgloss.spines['bottom'].set_position(('outward', 45))
    ax_avgloss.set_xlabel('Epochs')
    ax_avgloss.set_xlim(ax_loss.get_xlim())
    
    plt.title(f"Training loss for '{model_id}' | Total training time = {timedelta(seconds=training_time)}")

    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_models_training_losses(checkpoints:list[dict], labels:list[str]=None):  
    plt.figure()
    ax_loss = plt.subplot(1, 1, 1)
    # Set axis labels
    ax_loss.set_xlabel('Iterations')
    ax_loss.set_ylabel('Loss')
    
    if labels:
        if len(labels) != len(checkpoints):
            raise Exception("Length of 'labels' much match the length of given 'checkpoints'")
    
    max_epochs = -1
    iter_per_epoch = -1
    for i, checkpoint in enumerate(checkpoints):
        if "model_id" not in checkpoint:
            model_id = "temp"
        else: model_id = checkpoint["model_id"]
        
        loss_per_batch = checkpoint["loss_per_batch"]
        num_batch_losses = len(loss_per_batch)
        
        num_epochs = checkpoint["num_epochs"]
        if num_epochs > max_epochs: 
            max_epochs = num_epochs
            iter_per_epoch = len(loss_per_batch) // num_epochs
    
        # Create plot of total iterations (batch_size * num_epochs) against
        # loss for each batch
        ax_loss.plot(
            range(num_batch_losses), 
            loss_per_batch,
            label=f"{model_id if not labels else labels[i]} | Loss"
        )
        
        ax_loss.plot(
            np.convolve(loss_per_batch, 
                        np.ones(25,)/25, 
                        mode='valid'), 
            label=f"{model_id if not labels else labels[i]} | Running Average"
        )
        
    plt.legend()
        
    ax_avgloss = ax_loss.twiny()
    newlabel = list(range(max_epochs+1))
    
    newpos = [e*iter_per_epoch for e in newlabel]

    ax_avgloss.set_xticks(newpos[::num_epochs])
    ax_avgloss.set_xticklabels(newlabel[::num_epochs])

    ax_avgloss.xaxis.set_ticks_position('bottom')
    ax_avgloss.xaxis.set_label_position('bottom')
    ax_avgloss.spines['bottom'].set_position(('outward', 45))
    ax_avgloss.set_xlabel('Epochs')
    ax_avgloss.set_xlim(ax_loss.get_xlim())
    
    plt.title(f"Training loss")

    plt.grid()
    plt.tight_layout()
    plt.show()
    return
    
    # Set y-limit to 50% greater than maximum loss
    ax_loss.set_ylim([0, np.max(loss_per_batch)*1.5])
    
    # Calculate running average of batch losses and add to plot
    ax_loss.plot(np.convolve(loss_per_batch,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average')
    
    # Create legend
    ax_loss.legend()
    
    # Add new x-axis to display actual epoch range
    ax_avgloss = ax_loss.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax_avgloss.set_xticks(newpos[::num_epochs])
    ax_avgloss.set_xticklabels(newlabel[::num_epochs])

    ax_avgloss.xaxis.set_ticks_position('bottom')
    ax_avgloss.xaxis.set_label_position('bottom')
    ax_avgloss.spines['bottom'].set_position(('outward', 45))
    ax_avgloss.set_xlabel('Epochs')
    ax_avgloss.set_xlim(ax_loss.get_xlim())

    plt.title(f"Training loss for '{model_id}' | Total training time = {timedelta(seconds=training_time)}")

    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_downsampled_flowfield(params:dict, aerofoil_id:str, aoa:str):
    scaler = MinMaxScaler()
    
    original_flowfield_path = os.path.join(
        params["i/o"]["results_folder"],
        "fluent",
        aerofoil_id,
        f"{aoa}.csv"
    )
    downsampled_flowfield_path = os.path.join(
        params["i/o"]["datasets_folder"],
        "train",
        aerofoil_id,
        aoa,
        "flowfield.npy.gz"
    )
    removed_nodes_path = os.path.join(
        params["i/o"]["datasets_folder"],
        "train",
        aerofoil_id,
        aoa,
        "removed_nodes.npy.gz"
    )
    
    with open(original_flowfield_path, "r") as f:
        original_flowfield = np.array([l.rstrip().split()[1:] for l in f.readlines()[1:]], dtype=float)
    with gzip.open(downsampled_flowfield_path, "r") as f:
        downsampled_flowfield = np.load(f)
    with gzip.open(removed_nodes_path, "r") as f:
        removed_nodes = np.load(f)
    
    original_flowfield = scaler.fit_transform(original_flowfield)
    
    _, axs = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    axs[0].scatter(original_flowfield[:, 0], original_flowfield[:, 1], color="#000000", s=1)
    axs[0].grid()
    axs[0].axis([0.4, 0.525, 0.475, 0.525])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("Original Flow-field Nodes")
    axs[1].scatter(downsampled_flowfield[:, 0], downsampled_flowfield[:, 1], color="#000000", s=1)
    axs[1].scatter(removed_nodes[:, 0], removed_nodes[:, 1], color="#FF0000", s=1.5)
    axs[1].grid()
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title("Visualisation of removed nodes (Red)")
    axs[2].scatter(downsampled_flowfield[:, 0], downsampled_flowfield[:, 1], color="#000000", s=1)
    axs[2].grid()
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].set_title("Downsampled Flow-field Nodes")

    plt.suptitle(f"Downsampling of flow-field for '{aerofoil_id}'\nTotal node count : {len(original_flowfield)} -> {len(downsampled_flowfield)}")
    plt.show()
    
def plot_flowfield_contours(params:dict, checkpoint:dict, use_train_datasets:bool, channel:int):
    #region # ==== UNPACK PARAMETERS ==== #
    
    downsampled_nodes = params["preprocess"]["downsampled_nodes"]
    
    batch_size = params["training"]["batch_size"]
    
    datasets_folder = params["i/o"]["datasets_folder"]
    
    #endregion
    
    # Load model from checkpoint
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load and initialise datasets
    dataset = FlowField(datasets_folder, train=use_train_datasets, randomise=False)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    # Get and send model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Setup criterion to display loss of flowfield reconstruction
    criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        for batch_idx, (flow, aerofoil_id, aoa, _, aerofoil_coords) in enumerate(dataloader):
            # Send flowfield tensor to model and generate reconstructed field
            flow = flow.to(device)
            reconstructed_flow, _ = model(flow)
            losses = criterion(reconstructed_flow, flow.x)
            
            # Detach and convert data to local numpy arrays
            flow = flow.x.detach().cpu().numpy()
            flow = flow.reshape(batch_size, downsampled_nodes, flow.shape[1])
            reconstructed_flow = reconstructed_flow.detach().cpu().numpy()
            reconstructed_flow = reconstructed_flow.reshape(batch_size, downsampled_nodes, flow.shape[0])
            losses = losses.detach().cpu().numpy()
            losses = losses.reshape(batch_size, downsampled_nodes, flow.shape[0])
            
            # For each batch from dataloader, display flowfield contour of original and
            # reconstructed data
            for i in range(batch_size):
                base_coords = aerofoil_coords[0][i].numpy()
                flap_coords = aerofoil_coords[1][i].numpy()
                slat_coords = aerofoil_coords[2][i].numpy()
                
                base_polygon = Polygon(base_coords)
                flap_polygon = Polygon(flap_coords)
                slat_polygon = Polygon(slat_coords)
                
                triang = Triangulation(flow[i][:, 0], flow[i][:, 1])
                triang_points = np.column_stack((triang.x, triang.y))
                remove_mask = np.zeros(len(triang.triangles), dtype=bool)
                
                for j, triangle in enumerate(triang.triangles):
                    triangle_coords = triang_points[triangle]
                    
                    edges = [
                        LineString([triangle_coords[0], triangle_coords[1]]),
                        LineString([triangle_coords[1], triangle_coords[2]]),
                        LineString([triangle_coords[2], triangle_coords[0]])
                    ]
                    
                    for edge in edges:
                        if edge.intersects(base_polygon) or edge.intersects(flap_polygon) or edge.intersects(slat_polygon):
                            remove_mask[j] = True
                
                triang.set_mask(remove_mask)
                
                # Create subplots
                fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
                axs[0].tricontourf(triang, flow[i][:, channel], levels=100, cmap="jet")
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                axs[1].tricontourf(triang, flow[i][:, channel], levels=100, cmap="jet")
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                losses_contour = axs[2].tricontourf(triang, losses[i][:, channel], levels=100, cmap="jet")
                axs[2].set_xticks([])
                axs[2].set_yticks([])
                axs[0].title.set_text("Original Flowfield")
                axs[1].title.set_text("Reconstructed Flowfield")
                axs[2].title.set_text("Loss / Error")
                
                plt.colorbar(losses_contour, ax=axs[2])
                
                if channel == 0: quantity = "X-Coordinate"
                elif channel == 1: quantity = "Y-Coordinate"
                elif channel == 2: quantity = "Pressure Coefficient"
                elif channel == 3: quantity = "Mach Number"
                else: quantity = "Vorticity Magnitude"
                
                fig.suptitle(f"Flowfield reconstruction of '{aerofoil_id[i]}' at AoA = '{aoa[i]}' ({quantity}) \nTotal Mean Loss = {np.mean(losses[i][:, channel], axis=0):.4f}, Max Loss = {np.max(losses[i][:, channel], axis=0):.4f}")
                
                #plt.tight_layout()
                plt.show()

def plot_embedding_relationships(params:dict, checkpoint:dict, use_train_datasets:bool):
    #region # ==== UNPACK PARAMETERS ==== #
    
    datasets_folder = params["i/o"]["datasets_folder"]
    
    #endregion

    model = checkpoint["model"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    dataset = FlowField(datasets_folder, train=use_train_datasets, randomise=False)
    dataloader = DataLoader(dataset=dataset, batch_size=15, shuffle=False) # Batch size 15 for number of AoA entries
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for batch_idx, (flow, aerofoil_id, aoa, coeffs) in enumerate(dataloader):
            flow = flow.to(device)
            _, embedding = model(flow)
            embedding = embedding.detach().cpu().numpy()
            aoa = aoa.numpy()
            coeffs = coeffs.numpy()
            
            _, axs = plt.subplots(nrows=1, ncols=4, sharex=True)
            color = np.random.rand(3,)
            for i in range(embedding.shape[0]):
                axs[0].scatter(aoa[i], embedding[i][0], color=color)
                axs[0].grid()
                axs[0].set_xlabel("Angle of Attack (°)")
                axs[0].set_ylabel("Parameter 1")
                axs[1].scatter(aoa[i], embedding[i][1], color=color)
                axs[1].grid()
                axs[1].set_xlabel("Angle of Attack (°)")
                axs[1].set_ylabel("Parameter 2")
                axs[2].scatter(aoa[i], embedding[i][2], color=color)
                axs[2].grid()
                axs[2].set_xlabel("Angle of Attack (°)")
                axs[2].set_ylabel("Parameter 3")
                axs[3].scatter(aoa[i], coeffs[i][0], color=color)
                axs[3].grid()
                axs[3].set_xlabel("Angle of Attack (°)")
                axs[3].set_ylabel("Coefficient of Lift")
            plt.suptitle(f"{aerofoil_id[i]} | Relationships of latent parameters against AoA")
            plt.show()
            
def plot_refined_aoa_study(params:dict, checkpoint:dict):
    datasets_folder = os.path.join(params["i/o"]["datasets_folder"], "study")

    aerofoil1_aoas = np.concat((np.linspace(-5, 2.5, 4), np.linspace(5, 7.5, 10), np.linspace(10, 30, 9)))
    aerofoil23_aoas = np.concat((np.linspace(-5, 7.5, 6), np.linspace(10, 12.5, 10), np.linspace(15, 30, 7)))
    
    aoas = 23*[aerofoil1_aoas] + 46*[aerofoil23_aoas]
    
    dataset = FlowField(root_dir=datasets_folder, randomise=False, train=True, aoas=aoas)
    dataloader = DataLoader(dataset=dataset, batch_size=23, shuffle=False)
    
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for batch_idx, (flow, aerofoil_id, aoa, coeffs) in enumerate(dataloader):
            flow = flow.to(device)
            _, embedding = model(flow)
            embedding = embedding.detach().cpu().numpy()
            aoa = aoa.numpy()
            coeffs = coeffs.numpy()
            
            _, axs = plt.subplots(nrows=1, ncols=4, sharex=True)
            color = np.random.rand(3,)
            for i in range(embedding.shape[0]):
                axs[0].scatter(aoa[i], embedding[i][0], color=color)
                axs[0].grid()
                axs[0].set_xlabel("Angle of Attack (°)")
                axs[0].set_ylabel("Parameter 1")
                axs[1].scatter(aoa[i], embedding[i][1], color=color)
                axs[1].grid()
                axs[1].set_xlabel("Angle of Attack (°)")
                axs[1].set_ylabel("Parameter 2")
                axs[2].scatter(aoa[i], embedding[i][2], color=color)
                axs[2].grid()
                axs[2].set_xlabel("Angle of Attack (°)")
                axs[2].set_ylabel("Parameter 3")
                axs[3].scatter(aoa[i], coeffs[i][0], color=color)
                axs[3].grid()
                axs[3].set_xlabel("Angle of Attack (°)")
                axs[3].set_ylabel("Coefficient of Lift")
            plt.suptitle(f"{aerofoil_id[i]} | Relationships of latent parameters against AoA")
            plt.show()

def plot_coords(params:dict):
    # (10, -10) -> (1, 0)
    
    with open(r"C:\Users\honey\Documents\PROJECT\coordinates\NACA-0313-56566565202816-2207163230\base.txt", "r") as f:
        coords = np.array([l.rstrip().split()[1:] for l in f.readlines()[1:]], dtype=float)
        
    with open(r"C:\Users\honey\Documents\PROJECT\results\fluent\NACA-0313-56566565202816-2207163230\0.0.csv", "r") as f:
        flowfield = np.array([l.rstrip().split()[1:] for l in f.readlines()[1:]], dtype=float)
    
    N = len(coords)
    coords = np.concat((coords, flowfield[:, :2]))
    
    scaler = MinMaxScaler()
    coords = scaler.fit_transform(coords)
    flowfield = scaler.fit_transform(flowfield)
    
    coords = coords[:N]
    
    plt.scatter(flowfield[:, 0], flowfield[:, 1])
    plt.scatter(coords[:, 0], coords[:, 1])
    plt.show()

if __name__ == "__main__":
    params = load_parameters()
    checkpoint = fetch_checkpoint(params, "model_2025-03-27_15-06-59")
    plot_refined_aoa_study(params, checkpoint)
    #plot_flowfield_contours(params, checkpoint, True, 3)
    #plot_coords(params)
    #plot_downsampled_flowfield(params, "NACA-0313-56566565202816-2207163230", "-5.0")