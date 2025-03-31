import glob
import gzip
import logging
import os

import numpy as np
import torch

def fetch_checkpoint(params:dict, checkpoint:str, logger:logging.Logger=None):
    if logger: logger.info("Fetching checkpoint...")
    #region # ==== UNPACK PARAMETERS ==== #
    
    checkpoints_folder = params["i/o"]["checkpoints_folder"]
    
    #endregion
    
    checkpoint_root = os.path.join(checkpoints_folder, checkpoint)
    
    latest_checkpoint = glob.glob(os.path.join(checkpoint_root, "*.pt"))[-1]
    
    # Load checkpoint
    checkpoint = torch.load(latest_checkpoint, weights_only=False)

    if logger: logger.warning(f"Checkpoint successfully fetched! Statistics:\nEpochs: {checkpoint['num_epoch']}, Loss: {checkpoint['loss'].item()}")

    return checkpoint

def conv_output_size(input_size, kernel_size, stride, padding):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1

def conv_transpose_output_size(input_size, kernel_size, stride, padding, output_padding):
    return stride * (input_size - 1) + kernel_size - 2 * padding + output_padding

def combine_coefficients_array():
    with gzip.open(r"C:\Users\honey\Documents\PROJECT\datasets\study\train\NACA-0313-74556969223627-2207163230\coefficients1.npy.gz", "r") as f:
        aerofoil1_coeffs1 = np.load(f)
    with gzip.open(r"C:\Users\honey\Documents\PROJECT\datasets\study\train\NACA-0313-74556969223627-2207163230\coefficients2.npy.gz", "r") as f:
        aerofoil1_coeffs2 = np.load(f)
        
    with gzip.open(r"C:\Users\honey\Documents\PROJECT\datasets\study\train\NACA-2316-56566565202816-2207163230\coefficients1.npy.gz", "r") as f:
        aerofoil2_coeffs1 = np.load(f)
    with gzip.open(r"C:\Users\honey\Documents\PROJECT\datasets\study\train\NACA-2316-56566565202816-2207163230\coefficients2.npy.gz", "r") as f:
        aerofoil2_coeffs2 = np.load(f)
        
    with gzip.open(r"C:\Users\honey\Documents\PROJECT\datasets\study\train\NACA-2416-56566565202816-2207163230\coefficients1.npy.gz", "r") as f:
        aerofoil3_coeffs1 = np.load(f)
    with gzip.open(r"C:\Users\honey\Documents\PROJECT\datasets\study\train\NACA-2416-56566565202816-2207163230\coefficients2.npy.gz", "r") as f:
        aerofoil3_coeffs2 = np.load(f)
        
    aerofoil1_coeffs = np.concat((aerofoil1_coeffs2[:4], aerofoil1_coeffs1, aerofoil1_coeffs2[4:]))
    aerofoil2_coeffs = np.concat((aerofoil2_coeffs2[:6], aerofoil2_coeffs1, aerofoil2_coeffs2[6:]))
    aerofoil3_coeffs = np.concat((aerofoil3_coeffs2[:6], aerofoil3_coeffs1, aerofoil3_coeffs2[6:]))
    
    f = gzip.GzipFile(r"C:\Users\honey\Documents\PROJECT\datasets\study\train\NACA-0313-74556969223627-2207163230\coefficients.npy.gz", "wb")
    np.save(file=f, arr=aerofoil1_coeffs)
    f = gzip.GzipFile(r"C:\Users\honey\Documents\PROJECT\datasets\study\train\NACA-2316-56566565202816-2207163230\coefficients.npy.gz", "wb")
    np.save(file=f, arr=aerofoil2_coeffs)
    f = gzip.GzipFile(r"C:\Users\honey\Documents\PROJECT\datasets\study\train\NACA-2416-56566565202816-2207163230\coefficients.npy.gz", "wb")
    np.save(file=f, arr=aerofoil3_coeffs)

if __name__ == "__main__":
    combine_coefficients_array()