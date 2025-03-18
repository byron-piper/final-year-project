from datetime import datetime
import gzip
import logging
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import torch
from torch.utils.data import Dataset
from tsdownsample import MinMaxLTTBDownsampler

sys.path.append(r"C:\Users\honey\Documents\PROJECT\final-year-project")

from sample.helper import load_parameters

def _minmax_lttb_downsampling(flowfield:np.ndarray, nodes_out:int) -> tuple[np.ndarray, np.ndarray]:
    #region docstring
    """
    Downsamples flowfield nodes given in Numpy array (x, y, cp, v, ω) to reduced node count.
    
    Utilises the `tsdownsample` library: https://github.com/predict-idlab/tsdownsample
    
    Parameters
    ----------
    flowfield : np.ndarray
        Numpy array containing (x, y, cp, v, ω) flowfield quantities
    nodes_out : int
        Number of nodes to reduce flowfield to
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing Numpy arrays of downsampled flowfield and remove nodes respectively
    """
    #endregion
    
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

def _distance_based_downsampling(params:dict, flowfield:np.ndarray, threshold:float):
    coords_folder = os.path.join(params["i/o"]["coords_folder"], "NACA-0313-56566565202816-2207163230")
    
    with open(os.path.join(coords_folder, "base.txt"), "r") as f:
        base_coords = np.array([line.strip().split()[1:] for line in f.readlines()[1:]], dtype=float)
    with open(os.path.join(coords_folder, "flap.txt"), "r") as f:
        flap_coords = np.array([line.strip().split()[1:] for line in f.readlines()[1:]], dtype=float)
    with open(os.path.join(coords_folder, "slat.txt"), "r") as f:
        slat_coords = np.array([line.strip().split()[1:] for line in f.readlines()[1:]], dtype=float)
    
    combined_coords = np.concat((base_coords, flap_coords, slat_coords))
    
    tree = KDTree(combined_coords)
    
    distances, _ = tree.query(flowfield[:, :2])
    
    filtered_indicies = distances <= threshold
    other_indicies = distances > threshold
    
    filtered_flowfield = flowfield[filtered_indicies]
    other_flowfield = flowfield[other_indicies]
    
    return filtered_flowfield, other_flowfield

def _kmeans_clustering_downsampling(flowfield:np.ndarray, nodes_out:int, max_iters:int=100, batch_size:int=10000):
    device = torch.device("cuda")
    
    flowfield_tensor = torch.tensor(flowfield, dtype=torch.float16, device=device)
    xy = flowfield_tensor[:, :2]
    
    indices = torch.randperm(xy.shape[0])[:nodes_out]
    centroids = xy[indices].clone()
    
    for _ in range(max_iters):
        new_centroids = torch.zeros_like(centroids, device=device)
        counts = torch.zeros(nodes_out, device=device)

        for i in range(0, xy.shape[0], batch_size):
            batch = xy[i:i+batch_size]
            dists = torch.cdist(batch, centroids)
            labels = torch.argmin(dists, dim=1)
        
            for j in range(nodes_out):
                mask = labels == j
                if mask.any():
                    new_centroids[j] += batch[mask].sum(dim=0)
                    counts[j] += mask.sum()
        
        # Normalize centroids (avoid divide-by-zero issues)
        mask = counts > 0
        new_centroids[mask] /= counts[mask].unsqueeze(1)

        # Stop early if centroids don't change significantly
        if torch.allclose(new_centroids, centroids, atol=1e-4):
            break

        centroids = new_centroids.clone()

    # Compute flow variable averages for each cluster
    reduced_data = []
    for i in range(nodes_out):
        cluster_indices = (labels == i).nonzero(as_tuple=True)[0]
        cluster_points = flowfield_tensor[cluster_indices]

        if len(cluster_points) > 0:
            centroid = centroids[i].cpu().numpy().astype(np.float32)
            if flowfield.shape[1] > 2:  # If flow variables exist
                avg_flow_vars = cluster_points[:, 2:].mean(dim=0).cpu().numpy().astype(np.float32)
                centroid = np.concatenate((centroid, avg_flow_vars))

            reduced_data.append(centroid)
            
    return np.array(reduced_data)

def _dbscan_clustering_downsampling(flowfield:np.ndarray):
    eps = 0.005
    min_samples = 100000
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(flowfield)
    
    cluster_labels = db.labels_
    
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
    
    reduced_points = []
    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_points = flowfield[cluster_indices]

        # Compute centroid
        centroid = np.mean(cluster_points, axis=0)

        # Average flow variables if they exist
        if flowfield.shape[1] > 2:  # If more columns exist beyond X, Y
            avg_values = np.mean(flowfield[cluster_indices, 2:], axis=0)
            centroid = np.concatenate((centroid, avg_values))

        reduced_points.append(centroid)

    # Convert to NumPy array
    reduced_flowfield = np.array(reduced_points)
    
    return reduced_flowfield
 
def fast_interpolation(kept_nodes, removed_nodes):
    """
    Interpolates between downsampled (kept) nodes and removed nodes using fast linear interpolation.
    
    Args:
        kept_nodes (numpy.ndarray): Array of retained nodes after LTTB downsampling (N, D).
        removed_nodes (numpy.ndarray): Array of removed nodes (M, D).

    Returns:
        numpy.ndarray: Interpolated nodes (same shape as removed_nodes).
    """
    if removed_nodes.shape[0] == 0:
        return np.array([])  # No removed nodes, return empty array

    # Find the nearest two kept nodes for each removed node
    x_kept = kept_nodes[:, 0]  # X-coordinates of kept nodes
    x_removed = removed_nodes[:, 0]  # X-coordinates of removed nodes
    
    # Sort kept nodes by X to ensure correct interpolation
    sort_idx = np.argsort(x_kept)
    x_kept = x_kept[sort_idx]
    kept_nodes = kept_nodes[sort_idx]

    # Find left and right indices of kept nodes for interpolation
    left_idx = np.searchsorted(x_kept, x_removed, side='right') - 1
    right_idx = np.clip(left_idx + 1, 0, len(x_kept) - 1)

    # Linear interpolation: y = y1 + (y2 - y1) * ((x - x1) / (x2 - x1))
    x1, x2 = x_kept[left_idx], x_kept[right_idx]
    y1, y2 = kept_nodes[left_idx], kept_nodes[right_idx]

    # Avoid division by zero if x1 == x2
    ratio = np.where(x2 != x1, (x_removed - x1) / (x2 - x1), 0)
    interpolated_nodes = y1 + ratio[:, None] * (y2 - y1)

    return interpolated_nodes 

def generate_fluent_datasets(params:dict, compute_downsample_nodes:bool=False, visualise:bool=False) -> None:
    #region docstring
    """
    Generates `.npy.gz` datasets containing downsampled flowfields generated by Fluent simulations to
    datasets folder given by `params`. Downsampled node count can be calculated by obtaining the minimum
    of nodes from all result files found. `visualise` can be set to `True` to enable visualisation of the
    downsampling result.
    
    Parameters
    ----------
    params : dict
        Dictionary containing workflow parameters
    compute_downsample_nodes : bool
        Flag to toggle computation of downsampling size based on minimum nodes from results
    visualise : bool
        Flag to toggle visualisation of downsampling. If enabled, writing of the dataset disabled.
    
    Returns
    -------
    None
    """
    #endregion
    
    # Get I/O paths
    results_folder = params["i/o"]["results_folder"]
    datasets_folder = params["i/o"]["datasets_folder"]
    
    visualise = params["preprocess"]["visualise"]
    train_batch_size = params["preprocess"]["train_batch_size"]
    
    # Determine the number of nodes used to downsample from each flowfield to unify mesh node count
    logging.info("[-/-] : - | Settings downsampling size to default: 100,000 nodes.")
    nodes_out = 320**2
    if compute_downsample_nodes:
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
    elif params["preprocess"]["field_dim"] > 0:
        nodes_out = params["preprocess"]["field_dim"]**2
        logging.info(f"[-/-] : - | Downsampling flowfield nodes taken from parameters: {nodes_out}")
    
    #region # ==== DOWNSAMPLE FLOWFIELDS ==== #
    
    def visualise_downsampling(flowfield:np.ndarray, downsampled_flowfield:np.ndarray, removed_nodes:np.ndarray) -> None:
        #region docstring
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

        Returns
        -------
        None
        """
        #endregion
        
        fig, ax = plt.subplots(3, sharex=True, sharey=True)
        fig.suptitle(f"{aerofoil_id}: {len(lines)} -> {len(downsampled_flowfield)} nodes")
        
        ax[0].scatter(flowfield[:, 0], flowfield[:, 1], s=1)
        ax[0].grid()
        #ax[0].axis([-0.5, 1.25, -0.2, 0.2])
        ax[0].set_xlabel("x-coordinate (m)")
        ax[0].set_ylabel("y-coordinate (m)")
        
        ax[1].scatter(downsampled_flowfield[:, 0], downsampled_flowfield[:, 1], s=1)
        if removed_nodes is not None:
            ax[1].scatter(removed_nodes[:, 0], removed_nodes[:, 1], s=1, color='#ff0000')
        ax[1].grid()
        #ax[1].axis([-0.5, 1.25, -0.2, 0.2])
        ax[1].set_xlabel("x-coordinate (m)")
        ax[1].set_ylabel("y-coordinate (m)")

        ax[2].scatter(downsampled_flowfield[:, 0], downsampled_flowfield[:, 1], s=1)
        ax[2].grid()
        #ax[2].axis([-0.5, 1.25, -0.2, 0.2])
        ax[2].set_xlabel("x-coordinate (m)")
        ax[2].set_ylabel("y-coordinate (m)")
        plt.show()
    
    # Walk through each aerofoil folder in the results folder and downsample each .csv file to unify
    # number of nodes
    logging.info(f"[-/-] : - | Fetching all '.csv' result files from results folder: {results_folder}")
    completed_loops = 1
    # Get number of aerofoil results
    num_aerofoils = len(os.listdir(results_folder))
    removed_nodes = None
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
                downsampled_flowfield, removed_nodes = _minmax_lttb_downsampling(flowfield, nodes_out)
                logging.info(f"[{completed_loops}/{num_aerofoils*15}] : {aerofoil_id} | Downsampled flowfield from {len(lines)} to {len(downsampled_flowfield)} nodes...")
                
                # Either visualise the downsampling or save the downsampled data
                if not visualise:
                    if  completed_loops <= train_batch_size:
                        dest_folder = os.path.join(datasets_folder, "train", aerofoil_id)
                    else:
                        dest_folder = os.path.join(datasets_folder, "test", aerofoil_id)
                        
                    Path(dest_folder).mkdir(parents=True, exist_ok=True)
                    
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

class FlowFieldDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_info = []
        
        root_dir = os.path.join(root_dir, "train" if train else "test")
        
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path):
                continue
            for file in os.listdir(label_path):
                if file.endswith(".npy.gz"):
                    file_path = os.path.join(label_path, file)
                    aoa = float(file.replace(".npy.gz", ""))
                    self.data_info.append((file_path, aoa))
                    
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        file_path, angle = self.data_info[idx]
        
        with gzip.open(file_path, "r") as f:
            data = np.load(f)
            
        #data = (data - np.min(data)) / (np.max(data) - np.min(data))
           
        col_min = np.min(data, axis=0)
        col_max = np.max(data, axis=0)
        
        denom = col_max - col_min
        denom[denom == 0] = 1
        
        data = (data - col_min) / denom
        
        xy = data[:, :2]
        
        # v_channel = data[:, 3].astype(np.float32)
        
        #v_channel = (v_channel - np.min(v_channel)) / (np.max(v_channel) - np.min(v_channel))
        
        data = data.reshape(320, 320, 5)
        data = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)
        
        if self.transform:
            data = self.transform(data)
            
        return data, xy, angle

if __name__ == "__main__":
    parameters = load_parameters()
        
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