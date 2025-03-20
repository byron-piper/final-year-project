from datetime import datetime
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import torch
from torch.functional import F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as DL

from datasets import FlowField
from models import VAE, GCCN, GCCNLinear

sys.path.append(r"C:\Users\honey\Documents\PROJECT\final-year-project")

from sample.helper import load_parameters
from utils import fetch_model_dict
    
def train_model(params:dict, logger:logging.Logger, model_dict:dict, model_instance, save_model:bool=False):
    #region # ==== UNPACK PARAMETERS ==== #
    
    checkpoint = params["training"]["checkpoint"]
    num_epochs = params["training"]["num_epochs"]
    learning_rate = params["training"]["learning_rate"]
    weight_decay = params["training"]["weight_decay"]
    batch_size = params["training"]["batch_size"]
    logging_interval = params["training"]["logging_interval"]
    saving_interval = params["training"]["saving_interval"]
    
    datasets_folder = params["i/o"]["datasets_folder"]
    results_folder = os.path.join(params["i/o"]["results_folder"], "training")
    checkpoints_folder = params["i/o"]["checkpoints_folder"]
    
    #endregion
    
    #region # ==== SETUP DATASETS AND MODEL ==== #
    
    train_datasets = FlowFieldDataset(datasets_folder, train=True)
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_instance().to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CyclicLR(optimiser, base_lr=learning_rate, max_lr=0.01)
    
    logger.info(f"Setting up '{model_instance.__name__}' model on '{device}'.")
    logger.info(f"Training dataset loaded from '{datasets_folder}' of size '{len(train_loader)}' with batch size of '{batch_size}'")
    
    #endregion
    
    log_dict = {
        "train_loss_per_batch": [],
        "train_loss_per_epoch": []
    }
    
    loss_fn = F.mse_loss
    
    flow, xy, _ = train_datasets[0]
    flow = flow.view(1, 5, 320, 320)
    flow = flow.to(device)
    
    for epoch in range(5000):
        optimiser.zero_grad()
        output = model(flow)
        loss = loss_fn(output, flow)
        loss.backward()
        optimiser.step()
        #scheduler.step()
        print(f"Epoch {epoch}: Loss = {loss.item()}")
    
    model_dict = {
        "model": model,
        "epoch": num_epochs,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict()
    }
    
    flow = flow.view(-1, 102400, 5).detach().cpu().numpy()
    output = output.view(-1, 102400, 5).detach().cpu().numpy()
    
    x = xy[:, 0]
    y = xy[:, 1]
    
    triang = Triangulation(x, y)
    
    print(len(x), flow.shape)
    
    plt.subplot(1,2,1)
    plt.tricontourf(triang, flow[0][:, 3], levels=100, cmap="jet")
    plt.title("Input")

    plt.subplot(1,2,2)
    plt.tricontourf(triang, output[0][:, 3], levels=100, cmap="jet")
    plt.title("Reconstructed")
    plt.show()
    
    if checkpoint:
        checkpoint_path = os.path.join(checkpoints_folder, checkpoint)
    else:
        checkpoint_path = os.path.join(checkpoints_folder, "checkpoint.pt")
    logger.info(f"Saving checkpoint to '{checkpoint_path}'...")
    print(f"Saving checkpoint to '{checkpoint_path}'...")
    torch.save(model_dict, checkpoint_path)
    logger.info(f"Checkpoint saved.")
    print(f"Checkpoint saved.")
    
    return    

    loss_fn = F.mse_loss
    logger.info(f"Training model for '{num_epochs}' epochs...")
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (field, xy, _) in enumerate(train_loader):
            batch_item_start_time = time.time()
            field = field.view(batch_size, 5, 320, 320)
            field = field.to(device)
            recon = model(field)
            loss = loss_fn(recon, field)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            
            log_dict["train_loss_per_batch"].append(loss.item())
            
            if not batch_idx % logging_interval:
                logger.info(f"Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {loss:.4f} | Time Elapsed: {(time.time() - batch_item_start_time):.3f}s")
                print(f"Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {loss:.4f} | Time Elapsed: {(time.time() - batch_item_start_time):.3f}s")
        print(f"Time Elapsed: {(time.time() - start_time)/60:.4f} min")
    logger.info(f"Training complete, total training time: {(time.time() - start_time)/60:.4f} min")
    print(f"Training complete, total training time: {(time.time() - start_time)/60:.4f} min")
    
    model_dict = {
        "model": model,
        "epoch": num_epochs,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict()
    }
    
    if save_model:
        if checkpoint:
            checkpoint_path = os.path.join(checkpoints_folder, checkpoint)
        else:
            checkpoint_path = os.path.join(checkpoints_folder, "checkpoint.pt")
        logger.info(f"Saving checkpoint to '{checkpoint_path}'...")
        print(f"Saving checkpoint to '{checkpoint_path}'...")
        torch.save(model_dict, checkpoint_path)
        logger.info(f"Checkpoint saved.")
        print(f"Checkpoint saved.")
    
    return model_dict

def train_vae(params:dict, logger:logging.Logger, model_dict:dict, save_model:bool=False):
    log_dict = {
        "train_combined_loss_per_batch": [],
        "train_combined_loss_per_epoch": [],
        "train_reconstruction_loss_per_batch": [],
        "train_kl_loss_per_batch": []
    }
    
    datasets_folder = params["i/o"]["datasets_folder"]
    checkpoints_folder = params["i/o"]["checkpoints_folder"]
    
    batch_size = params["training"]["batch_size"]
    learning_rate = params["training"]["learning_rate"]
    weight_decay = params["training"]["weight_decay"]
    num_epochs = params["training"]["num_epochs"]
    logging_interval = params["training"]["logging_interval"]
    
    train_datasets = FlowFieldDataset(datasets_folder, train=True)
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=5)
    
    loss_fn = F.mse_loss
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VAE().to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CyclicLR(optimiser, base_lr=learning_rate, max_lr=0.1)
    
    logger.info(f"Setting up '{VAE.__name__}' model on '{device}'.")
    logger.info(f"Training dataset loaded from '{datasets_folder}' of size '{len(train_loader)}' with batch size of '{batch_size}'")

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _, _) in enumerate(train_loader):
            #features = features.view(batch_size, 1, 320, 320)
            features = features.to(device)

            # FORWARD AND BACK PROP
            encoded, z_mean, z_log_var, decoded = model(features)
            
            # total loss = reconstruction loss + KL divergence
            #kl_divergence = (0.5 * (z_mean**2 + 
            #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var), 
                                      axis=1) # sum over latent dimension

            batchsize = kl_div.size(0)
            kl_div = kl_div.mean() # average over batch dimension
    
            pixelwise = loss_fn(decoded, features, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean() # average over batch dimension
            
            loss = 1*pixelwise + kl_div
            
            optimiser.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimiser.step()
            scheduler.step()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    model_dict = {
        "model": model,
        "epoch": num_epochs,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict()
    }
    
    if save_model:
        checkpoint_path = os.path.join(checkpoints_folder, "vae.pt")
        torch.save(model_dict, checkpoint_path)

def train_gccn(params:dict, logger:logging.Logger, model_dict:dict, save_model:bool=False):
    #region # ==== UNPACK PARAMETERS ==== #
    
    checkpoint = params["training"]["checkpoint"]
    num_epochs = params["training"]["num_epochs"]
    learning_rate = params["training"]["learning_rate"]
    weight_decay = params["training"]["weight_decay"]
    batch_size = params["training"]["batch_size"]
    logging_interval = params["training"]["logging_interval"]
    saving_interval = params["training"]["saving_interval"]
    
    datasets_folder = params["i/o"]["datasets_folder"]
    results_folder = os.path.join(params["i/o"]["results_folder"], "training")
    checkpoints_folder = params["i/o"]["checkpoints_folder"]
    
    #endregion
    
    #region # ==== SETUP DATASETS AND MODEL ==== #
    
    train_datasets = FlowField(datasets_folder, train=True)
    
    train_loader = DL(dataset=train_datasets, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCCNLinear(in_channels=3).to(device)
    model.train()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CyclicLR(optimiser, base_lr=learning_rate, max_lr=0.01)
    
    logger.info(f"Setting up '{GCCN.__name__}' model on '{device}'.")
    logger.info(f"Training dataset loaded from '{datasets_folder}' of size '{len(train_loader)}' with batch size of '{batch_size}'")
    
    #endregion
    
    log_dict = {
        "train_loss_per_batch": [],
        "train_loss_per_epoch": []
    }
    
    loss_fn = F.mse_loss
    logger.info(f"Training model for '{num_epochs}' epochs...")
    start_time = time.time()
    for epoch in range(num_epochs):
        for batch_idx, (flow, _) in enumerate(train_loader):
            batch_item_start_time = time.time()
            flow = flow.to(device)
            optimiser.zero_grad()
            output = model(flow)
            loss = loss_fn(output, flow.x)
            loss.backward()
            optimiser.step()
            scheduler.step()
            
            log_dict["train_loss_per_batch"].append(loss.item())
            
            if not batch_idx % logging_interval:
                logger.info(f"Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {loss:.4f} | Time Elapsed: {(time.time() - batch_item_start_time):.3f}s")
                print(f"Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {loss:.4f} | Time Elapsed: {(time.time() - batch_item_start_time):.3f}s")
        print(f"Time Elapsed: {(time.time() - start_time)/60:.4f} min")
    logger.info(f"Training complete, total training time: {(time.time() - start_time)/60:.4f} min")
    print(f"Training complete, total training time: {(time.time() - start_time)/60:.4f} min")

    
    model_dict = {
        "model": model,
        "epoch": num_epochs,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict()
    }
    
    if save_model:
        if checkpoint:
            checkpoint_path = os.path.join(checkpoints_folder, checkpoint)
        else:
            checkpoint_path = os.path.join(checkpoints_folder, "checkpoint.pt")
        logger.info(f"Saving checkpoint to '{checkpoint_path}'...")
        print(f"Saving checkpoint to '{checkpoint_path}'...")
        torch.save(model_dict, checkpoint_path)
        logger.info(f"Checkpoint saved.")
        print(f"Checkpoint saved.")
    
    return model_dict

if __name__ == "__main__":
    params = load_parameters()
    
    #region # ==== UNPACK PARAMETERS ==== #
    
    checkpoint = params["training"]["checkpoint"]
    
    # I/O
    logs_folder = os.path.join(params["i/o"]["logs_folder"], "training")
    
    #endregion
    
    if not os.path.exists(logs_folder):
        os.mkdir(logs_folder)
        
    log_filename = os.path.join(logs_folder,
                                datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log"))
    
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    
    logger = logging.getLogger("basic_logger")
    
    model_dict = fetch_model_dict(params, checkpoint, logger)
    model_dict = train_gccn(params, logger, model_dict, save_model=True)