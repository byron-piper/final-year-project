from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import torch
from torch.functional import F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as DL

from datasets import FlowField
from models import GCNLinear

sys.path.append(r"C:\Users\honey\Documents\PROJECT\final-year-project")

from sample.helper import load_parameters
from utils import fetch_checkpoint
    
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

def train_gcn(params:dict, logger:logging.Logger, checkpoint:dict={}):
    #region # ==== UNPACK PARAMETERS ==== #
    
    num_epochs = params["training"]["num_epochs"]
    learning_rate = params["training"]["learning_rate"]
    weight_decay = params["training"]["weight_decay"]
    batch_size = params["training"]["batch_size"]
    logging_interval = params["training"]["logging_interval"]
    saving_interval = params["training"]["saving_interval"]
    save_model = params["training"]["save_model"]
    
    datasets_folder = params["i/o"]["datasets_folder"]
    checkpoints_folder = params["i/o"]["checkpoints_folder"]
    
    #endregion
    
    # Remove all empty checkpoint folders
    for current_dir, subdirs, files in os.walk(checkpoints_folder, topdown=False):
        if not any(files):
            os.rmdir(current_dir)
    
    #region # ==== SETUP DATASETS AND MODEL ==== #
    
    train_dataset = FlowField(datasets_folder, train=True, randomise=True)
    # Get `in_channels` by fetching column length of input node matrix
    train_loader = DL(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If a `model_dict` is not given, create new model, optimiser and learning rate scheduler
    if not checkpoint:
        #model = GCCNLinear(in_channels=in_channels, dropout_p=0.3).to(device)
        model = VGAE()
        model.train()
    
        # Set up optimiser and learning rate scheduler to help with plateauing
        optimiser = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimiser, base_lr=learning_rate, max_lr=0.1)
        
        curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_id = f"model_{curr_time}"
        
        if saving_interval > 0 and saving_interval <= num_epochs:
            checkpoint_folder = os.path.join(checkpoints_folder, model_id)
            Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_id": model_id,
            "model": model,
            "optimiser": optimiser,
            "lr_scheduler": lr_scheduler,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "num_epochs": 0,
            "batch_size": batch_size,
            "curr_loss": None,
            "loss_per_batch": [],
            "loss_per_epoch": [],
            "training_time": 0
        }
    else:
        model = checkpoint["model"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser = checkpoint["optimiser"]
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        lr_scheduler = checkpoint["lr_scheduler"]
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        
        checkpoint_folder = os.path.join(checkpoints_folder, checkpoint["model_id"])
    
    logger.info(f"Setting up '{GCNLinear.__name__}' model on '{device}'.")
    logger.info(f"Training dataset loaded from '{datasets_folder}' of size '{len(train_loader)}' with batch size of '{batch_size}'")
    
    #endregion
    
    loss_fn = F.mse_loss
    logger.info(f"Training model for '{num_epochs}' epochs...")
    
    cumulative_time = 0
    completed_epochs = 1
    model = model.to(device)
    training_start_time = datetime.now()
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        for batch_idx, (flow, _, _, _) in enumerate(train_loader):
            batch_start_time = datetime.now()
            flow = flow.to(device)
            optimiser.zero_grad()      
            flow_recon, _, _ = model(flow)
            
            loss = loss_fn(flow_recon, flow.x)
            loss.backward()
            optimiser.step()
            
            checkpoint["loss_per_batch"].append(loss.item())
            
            batch_end_time = datetime.now()
            batch_time_elapsed = batch_end_time - batch_start_time
            
            if not batch_idx % logging_interval:
                logger.info(f"Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {loss:.4f} | Time Elapsed: {batch_time_elapsed.total_seconds():.3f}s")
                print(f"Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {loss:.4f} | Time Elapsed: {batch_time_elapsed.total_seconds():.3f}s")
        
        epoch_end_time = datetime.now()
        epoch_time_elapsed = epoch_end_time - epoch_start_time
        cumulative_time += epoch_time_elapsed.seconds
        checkpoint["training_time"] += cumulative_time
        epochs_remaining = num_epochs - completed_epochs
        avg_time_per_epoch = cumulative_time / (completed_epochs)
        estimated_time_remaining = timedelta(seconds=(avg_time_per_epoch * epochs_remaining))
        
        checkpoint["loss_per_epoch"].append(loss.item())
        checkpoint["loss"] = loss
        
        print(f"Time elapsed over epoch: {epoch_time_elapsed}")
        print(f"Estimated time remaining: {estimated_time_remaining}")
        logger.info(f"Time elapsed over epoch: {epoch_time_elapsed}")
        logger.info(f"Estimated time remaining: {estimated_time_remaining}")
        
        logger.info(f"Stepping learning rate scheduler. New learning rate: {lr_scheduler.get_last_lr()}")
        lr_scheduler.step()
        
        if saving_interval != 0 and (epoch+1) % saving_interval == 0 and save_model:
            checkpoint_idx = checkpoint["num_epochs"]+1
            checkpoint_path = os.path.join(checkpoint_folder, f"checkpoint_{checkpoint_idx}.pt")
            
            # Update `checkpoint`
            
            checkpoint["index"] = checkpoint_idx
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["optimiser_state_dict"] = optimiser.state_dict()
            checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
            checkpoint["curr_loss"] = loss
            
            logger.info(f"Saving checkpoint '{os.path.basename(checkpoint_path)}' to '{checkpoint_folder}'...")
            print(f"Saving checkpoint '{os.path.basename(checkpoint_path)}' to '{checkpoint_folder}'...")
            torch.save(checkpoint, checkpoint_path)

        checkpoint["num_epochs"] += 1
        completed_epochs += 1
        
    logger.info(f"Training complete, total training time: {datetime.now() - training_start_time}")
    print(f"Training complete, total training time: {datetime.now() - training_start_time}")
    
    return checkpoint

def train_vgae(params:dict, logger:logging.Logger, checkpoint:dict={}):
    #region # ==== UNPACK PARAMETERS ==== #
    
    num_epochs = params["training"]["num_epochs"]
    learning_rate = params["training"]["learning_rate"]
    weight_decay = params["training"]["weight_decay"]
    batch_size = params["training"]["batch_size"]
    logging_interval = params["training"]["logging_interval"]
    saving_interval = params["training"]["saving_interval"]
    save_model = params["training"]["save_model"]
    
    in_channels = params["model"]["in_channels"]
    hid_channels = params["model"]["hid_channels"]
    emb_channels = params["model"]["emb_channels"]
    
    datasets_folder = params["i/o"]["datasets_folder"]
    checkpoints_folder = params["i/o"]["checkpoints_folder"]
    
    #endregion
    
    # Remove all empty checkpoint folders
    for current_dir, subdirs, files in os.walk(checkpoints_folder, topdown=False):
        for subdir in subdirs:
            folder_path = os.path.join(checkpoints_folder, subdir)
            if not os.listdir(folder_path):
                os.rmdir(current_dir)
    
    #region # ==== SETUP DATASETS AND MODEL ==== #
    
    train_dataset = FlowField(datasets_folder, train=True, randomise=True)
    # Get `in_channels` by fetching column length of input node matrix
    train_loader = DL(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If a `model_dict` is not given, create new model, optimiser and learning rate scheduler
    if not checkpoint:
        #model = GCCNLinear(in_channels=in_channels, dropout_p=0.3).to(device)
        model = GCNLinear(in_channels=in_channels, hid_channels=hid_channels, emb_channels=emb_channels)
        model.train()
    
        # Set up optimiser and learning rate scheduler to help with plateauing
        optimiser = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimiser, base_lr=learning_rate, max_lr=0.1)
        
        curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_id = f"model_{curr_time}"
        
        if saving_interval > 0 and saving_interval <= num_epochs:
            checkpoint_folder = os.path.join(checkpoints_folder, model_id)
            Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_id": model_id,
            "model": model,
            "optimiser": optimiser,
            "lr_scheduler": lr_scheduler,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "num_epochs": 0,
            "batch_size": batch_size,
            "curr_loss": None,
            "loss_per_batch": [],
            "loss_per_epoch": [],
            "training_time": 0
        }
    else:
        model = checkpoint["model"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser = checkpoint["optimiser"]
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        lr_scheduler = checkpoint["lr_scheduler"]
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        
        checkpoint_folder = os.path.join(checkpoints_folder, checkpoint["model_id"])
    
    in_channels, hid_channels, emb_channels = model.get_channels()
    
    # Create a metadata dictionary to be saved alongside checkpoints to provide
    # context for each model
    checkpoint_metadata = {
        "model": GCNLinear.__name__,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "in_channels": in_channels,
        "hid_channels": hid_channels,
        "emb_channels": emb_channels
    }
    
    logger.info(f"Setting up '{GCNLinear.__name__}' model on '{device}'.")
    logger.info(f"Training dataset loaded from '{datasets_folder}' of size '{len(train_loader)}' with batch size of '{batch_size}'")
    
    #endregion
    
    loss_fn = F.mse_loss
    logger.info(f"Training model for '{num_epochs}' epochs...")
    
    cumulative_time = 0
    completed_epochs = 1
    model = model.to(device)
    training_start_time = datetime.now()
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        for batch_idx, (flow, _, _, _) in enumerate(train_loader):
            batch_start_time = datetime.now()
            flow = flow.to(device)
            optimiser.zero_grad()
            flow_recon, _ = model(flow)
            loss = loss_fn(flow_recon, flow.x)
            loss.backward()
            optimiser.step()
            
            checkpoint["loss_per_batch"].append(loss.item())
            
            batch_end_time = datetime.now()
            batch_time_elapsed = batch_end_time - batch_start_time
            
            if not batch_idx % logging_interval:
                logger.info(f"Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {loss:.4f} | Time Elapsed: {batch_time_elapsed.total_seconds():.3f}s")
                print(f"Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {loss:.4f} | Time Elapsed: {batch_time_elapsed.total_seconds():.3f}s")
        
        epoch_end_time = datetime.now()
        epoch_time_elapsed = epoch_end_time - epoch_start_time
        cumulative_time += epoch_time_elapsed.seconds
        checkpoint["training_time"] += cumulative_time
        epochs_remaining = num_epochs - completed_epochs
        avg_time_per_epoch = cumulative_time / (completed_epochs)
        estimated_time_remaining = timedelta(seconds=(avg_time_per_epoch * epochs_remaining))
        
        checkpoint["loss_per_epoch"].append(loss.item())
        checkpoint["loss"] = loss
        
        print(f"Time elapsed over epoch: {epoch_time_elapsed}")
        print(f"Estimated time remaining: {estimated_time_remaining}")
        logger.info(f"Time elapsed over epoch: {epoch_time_elapsed}")
        logger.info(f"Estimated time remaining: {estimated_time_remaining}")
        
        logger.info(f"Stepping learning rate scheduler. New learning rate: {lr_scheduler.get_last_lr()}")
        lr_scheduler.step()
        
        if saving_interval != 0 and (epoch+1) % saving_interval == 0 and save_model:
            checkpoint_idx = checkpoint["num_epochs"]+1
            checkpoint_path = os.path.join(checkpoint_folder, f"checkpoint_{checkpoint_idx}.pt")
            
            # Update `checkpoint`
            
            checkpoint["index"] = checkpoint_idx
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["optimiser_state_dict"] = optimiser.state_dict()
            checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
            checkpoint["curr_loss"] = loss
            
            logger.info(f"Saving checkpoint '{os.path.basename(checkpoint_path)}' to '{checkpoint_folder}'...")
            print(f"Saving checkpoint '{os.path.basename(checkpoint_path)}' to '{checkpoint_folder}'...")
            torch.save(checkpoint, checkpoint_path)
            
            # Save checkpoint metadata
            metadata_path = os.path.join(checkpoint_folder, f"metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(checkpoint_metadata, f, indent=2)

        checkpoint["num_epochs"] += 1
        completed_epochs += 1
        
    logger.info(f"Training complete, total training time: {datetime.now() - training_start_time}")
    print(f"Training complete, total training time: {datetime.now() - training_start_time}")
    
    return checkpoint


if __name__ == "__main__":
    params = load_parameters()
    
    #region # ==== UNPACK PARAMETERS ==== #
    
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
    
    #checkpoint = fetch_checkpoint(params, "model_2025-03-21_17-54-25", logger)
    checkpoint = train_vgae(params, logger)