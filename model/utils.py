import logging
import os

import torch

def fetch_model_dict(params:dict, checkpoint:str, logger:logging.Logger=None):
    if logger: logger.info("Fetching model dict...")
    #region # ==== UNPACK PARAMETERS ==== #
    
    checkpoints_folder = params["i/o"]["checkpoints_folder"]
    
    #endregion
    
    if not checkpoint:
        if logger: logger.warning("Checkpoint flag is null! No model dict fetched...")
        return {}
    
    checkpoint_path = os.path.join(checkpoints_folder, checkpoint)
    
    # Load checkpoint
    checkpoint_dict = torch.load(checkpoint_path, weights_only=False)

    if logger: logger.warning(f"Model dict successfully fetched! Statistics:\nEpochs: {checkpoint_dict['epoch']}, Loss: {checkpoint_dict['loss']}")

    return checkpoint_dict

def conv_output_size(input_size, kernel_size, stride, padding):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1

def conv_transpose_output_size(input_size, kernel_size, stride, padding, output_padding):
    return stride * (input_size - 1) + kernel_size - 2 * padding + output_padding

if __name__ == "__main__":
    print(conv_transpose_output_size(input_size=80, kernel_size=3, stride=2, padding=1, output_padding=1))