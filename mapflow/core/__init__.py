import os
import numpy as np
import torch
from .prof_marker import prof_marker

def load_weights(weights_dir, dtype=torch.bfloat16, layer_list=None):
    weights = {}
    for filename in os.listdir(weights_dir):
        # translator_layer_3.pt
        if filename.startswith("translator_layer_"):
            layer_idx = int(filename.split(".")[0].split("_")[-1])
            weight_path = os.path.join(weights_dir, filename)
            weight_tensor = torch.load(weight_path, map_location='cuda')["weight"]
            print(f"Loading weight for layer {layer_idx} from {weight_path}, shape: {weight_tensor.shape}")
            weight_softmax = torch.softmax(weight_tensor, dim=0).to(dtype)
            weights[layer_idx] = weight_softmax
    if layer_list is not None:
        layer_offset = min(layer_list) - min(weights.keys())
        weights = {layer_idx: weights[layer_idx - layer_offset] for layer_idx in layer_list}

    return weights
