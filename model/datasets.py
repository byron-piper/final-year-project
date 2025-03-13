from datetime import datetime
import gzip
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from tsdownsample import MinMaxLTTBDownsampler

def _downsample_flowfield(flowfield:np.ndarray, nodes_out:int) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsamples flowfield nodes to given node out count.
    
    Utilises the `tsdownsample` library: https://github.com/predict-idlab/tsdownsample
    
    Parameters
    ----------
    flowfield : np.ndarray
        Numpy array containing (x, y, pressure-coefficient, velocity-magnitude, vorticity-mag)
    nodes_out : int
        Number of nodes to keep in the flowfield
    """
    
    # Extract (x, y) coordinates from flowfield and set as contiguous arrays
    x = np.ascontiguousarray(flowfield[:, 0])
    y = np.ascontiguousarray(flowfield[:, 1])
    
    # Apply downsampling algorithm
    downsampled_indicies = MinMaxLTTBDownsampler().downsample(x, y, n_out=nodes_out, parallel=True)
    
    # Create a new 2D array containing downsampled (x, y) coordinates
    downsampled_coordinates = np.column_stack((x[downsampled_indicies], y[downsampled_indicies]))
    
    # Create a mask for flowfield array that contains the downsampled (x, y) coordinates
    mask = np.all(np.isin(flowfield[:, :2], downsampled_coordinates), axis=1)
    
    downsampled_flowfield = flowfield[mask]
    removed_nodes = flowfield[~mask]
    
    while(len(downsampled_flowfield) > nodes_out):
        downsampled_flowfield = np.delete(downsampled_flowfield, 0, axis=0)
    
    # Return both the downsampled flowfield array and the removed nodes
    return downsampled_flowfield, removed_nodes

def generate_fluent_datasets(params:dict, compute_sample_size:bool=False, visualise:bool=False):
    """
    Generates dataset from Fluent flowfield results.
    
    TODO: ADD DESCRIPTION
    
    Parameters
    ----------
    params : dict
        Dictionary containing workflow parameters
    compute_sample_size : bool
        Flag to toggle computation of downsampling size based on minimum nodes from results
    visualise : bool
        Flag to toggle visualisation of downsampling. If enabled, writing of the dataset disabled.
    """
    
    # Get I/O paths
    results_folder = params["i/o"]["results_folder"]
    datasets_folder = params["i/o"]["datasets_folder"]
    
    # Determine the number of nodes used to downsample from each flowfield to unify mesh node count
    logging.info("[-/-] : - | Settings downsampling size to default: 100,000 nodes.")
    nodes_out = 100000
    if compute_sample_size:
        logging.info("[-/-] : - | Computing number of downsampled flowfield nodes...")
        #region # ==== COMPUTE SAMPLE SIZE IN FLOW-FIELD RESULTS ==== #
    
        # Each flowfield mesh generated for each aerofoil have differing number of mesh nodes.
        # Therefore, the effective "image" loaded into the VAE will not be fixed. To recify this,
        # all flowfields are downsampled based on the minimum number of nodes from all flowfields
        # to achieve equal sizes.
        # 
        # Min mesh nodes from results: 360925
        
        #sample_size = np.inf if sample_size_override == -1 else sample_size_override
        for root, _, files in os.walk(results_folder):
            csv_files = [os.path.join(root, f) for f in files if f.split(".")[-1] == "csv"]
            if csv_files:
                # Get first '.csv' as each AoA have equal number of mesh nodes
                first_csv_file = csv_files[0]
                with open(first_csv_file, "r") as file:
                    node_count = len(file.readlines())
                    if nodes_out > node_count:
                        nodes_out = node_count

        logging.info(f"[-/-] : - | Calculated number of flowfield nodes: {nodes_out}")
    #endregion
    elif params["preprocess"]["downsampling"] > 0:
        nodes_out = params["preprocess"]["downsampling"]
        logging.info(f"[-/-] : - | Downsampling flowfield nodes taken from parameters: {nodes_out}")
    
    #region # ==== DOWNSAMPLE FLOWFIELDS ==== #
    
    def visualise_downsampling(flowfield:np.ndarray, downsampled_flowfield:np.ndarray, removed_nodes:np.ndarray):
        """
        Visualises the original flowfield and the downsampled flowfields along with the corresponding removed nodes.
        
        Parameters
        ----------
        flowfield : np.ndarray
            Numpy array containing the original flowfield values
        downsampled_flowfield : np.ndarray
            Numpy array containing the downsampled flowfield values
        removed_nodes : np.ndarray
            Numpy array containing the removed flowfield nodes

        """
        fig, ax = plt.subplots(3, sharex=True, sharey=True)
        fig.suptitle(f"{aerofoil_id}: {len(lines)} -> {len(downsampled_flowfield)} nodes")
        
        ax[0].scatter(flowfield[:, 0], flowfield[:, 1], s=1)
        ax[0].grid()
        ax[0].axis([-0.5, 1.25, -0.2, 0.2])
        ax[0].set_xlabel("x-coordinate (m)")
        ax[0].set_ylabel("y-coordinate (m)")
        
        ax[1].scatter(downsampled_flowfield[:, 0], downsampled_flowfield[:, 1], s=1)
        ax[1].scatter(removed_nodes[:, 0], removed_nodes[:, 1], s=1, color='#ff0000')
        ax[1].grid()
        ax[1].axis([-0.5, 1.25, -0.2, 0.2])
        ax[1].set_xlabel("x-coordinate (m)")
        ax[1].set_ylabel("y-coordinate (m)")

        ax[2].scatter(downsampled_flowfield[:, 0], downsampled_flowfield[:, 1], s=1)
        ax[2].grid()
        ax[2].axis([-0.5, 1.25, -0.2, 0.2])
        ax[2].set_xlabel("x-coordinate (m)")
        ax[2].set_ylabel("y-coordinate (m)")
        plt.show()
    
    # Walk through each aerofoil folder in the results folder and downsample each .csv file to unify
    # number of nodes
    logging.info(f"[-/-] : - | Fetching all '.csv' result files from results folder: {results_folder}")
    completed_loops = 1
    # Get number of aerofoil results
    num_aerofoils = len(os.listdir(results_folder))
    for root, _, files in os.walk(results_folder):
        aerofoil_id = os.path.basename(root)
        csv_files_list = [os.path.join(root, f) for f in files if f.split(".")[-1] == "csv"]
        if csv_files_list:
            logging.info(f"[{completed_loops}/{num_aerofoils*15}] : {aerofoil_id} | {len(csv_files_list)} result files found.")
            for csv_file in csv_files_list:
                # Extract AoA from '.csv' filename
                aoa = ".".join(os.path.basename(csv_file).split(".")[:-1])
                # Read '.csv' file and convert contents to numpy array
                with open(csv_file, "r") as f:
                    lines = [line.strip().split()[1:] for line in f.readlines()[1:]]
                flowfield = np.array(lines, dtype=float)            
                # Downsample flowfield data
                downsampled_flowfield, removed_nodes = _downsample_flowfield(flowfield, nodes_out)
                logging.info(f"[{completed_loops}/{num_aerofoils*15}] : {aerofoil_id} | Downsampled flowfield from {len(lines)} to {len(downsampled_flowfield)} nodes...")
                
                # Either visualise the downsampling or save the downsampled data
                if not visualise:           
                    # Save to datasets folder
                    dest_folder = os.path.join(datasets_folder, aerofoil_id)
                    if not os.path.exists(dest_folder):
                        os.mkdir(dest_folder)
                    out_file_name = os.path.join(dest_folder, f"{aoa}.npy.gz")
                    logging.info(f"[{completed_loops}/{num_aerofoils*15}] : {aerofoil_id} | Saving flowfield data to '{out_file_name}'...")
                    f = gzip.GzipFile(out_file_name, "w")
                    np.save(file=f, arr=downsampled_flowfield)
                completed_loops += 1
            # Visualise flowfield nodes
            if visualise:
                visualise_downsampling(flowfield, downsampled_flowfield, removed_nodes)
                
    #endregion

class FluentDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []
        
        # Recursively find all .npy.gz files
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".npy.gz"):
                    self.file_list.append(os.path.join(subdir, file))
                    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # Load the compressed numpy array
        with gzip.open(file_path, 'r') as f:
            data = np.load(f)  # Load as numpy array

        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        H, W = 256, 256
        
        data = data.reshape(-1, 5, H, W)

        # Convert to PyTorch tensor
        tensor_data = torch.tensor(data, dtype=torch.float32)

        # Ensure correct shape (C, H, W)
        if tensor_data.ndim == 2:  
            tensor_data = tensor_data.unsqueeze(0)  # Add channel dimension if missing

        if self.transform:
            tensor_data = self.transform(tensor_data)

        return tensor_data  # Autoencoder input = target (unsupervised learning)

if __name__ == "__main__":
    with open("parameters.json", "r") as f:
        parameters = json.load(f)
        
    datasets_log_folder = os.path.join(parameters["i/o"]["logs_folder"], "datasets")
    
    if not os.path.exists(datasets_log_folder):
        os.mkdir(datasets_log_folder)
        
    log_filename = os.path.join(datasets_log_folder,
        datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log")
    )

    # Configure logging
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    
    generate_fluent_datasets(parameters)