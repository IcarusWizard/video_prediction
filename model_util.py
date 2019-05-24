import torch
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