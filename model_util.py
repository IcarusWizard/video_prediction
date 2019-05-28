import torch
from torch.functional import F
import numpy as np

import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def mse_loss(observations, predicted_observations):
    T = observations.shape[0]
    return torch.sum((observations[1 : T] - predicted_observations[0 : T -1 ]) ** 2) / 2.0

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(model, filename, eval_mode=False):
    model.load_state_dict(torch.load(filename))
    if eval_mode:
        model.eval()

def apply_cdna_kernel(image, kernel):
    """
        Inputs:
            image -> tensor[B, C, H, W]
            kernal -> tensor[B, N, K, K]
        Outputs:
            new_image -> tensor[B, N, C, H, W]
    """
    batch_size = image.shape[0]
    image_channel = image.shape[1]
    image_height = image.shape[2]
    image_width = image.shape[3]
    num_kernel = kernel.shape[1]
    kernel_size = kernel.shape[2]
    padding = kernel_size // 2

    _image = image.transpose(0, 1) # [C, B, H, W]
    _kernel = kernel.view(batch_size * num_kernel, 1, kernel_size, kernel_size) # [B * N, 1, K, K]

    output = F.conv2d(_image, _kernel, stride=1, padding=padding, groups=batch_size) # [C, B * N, H, W]

    output = output.view(image_channel, batch_size, num_kernel, image_height, image_width) # [C, B, N, H, W]

    return output.permute(1, 2, 0, 3, 4) # [B, N, C, H, W]




