import torch
import numpy as np

def preprocess_obs(obs_dict):
    image = obs_dict["image"]  # shape: (7, 7, 3)
    image = image.astype(np.float32) / 10.0  # normalize categorical encoding
    flat = image.flatten()
    return torch.tensor(flat).unsqueeze(0)  # shape: [1, obs_dim]