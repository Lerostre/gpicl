import pandas as pd
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from transformers import GPT2Config, GPT2ForTokenClassification
from transformers import get_cosine_schedule_with_warmup

from torchvision.transforms import v2

# bricks fpr params declaration
optim_params = {    
    "optimizer": optim.Adam,
    "optimizer_kwargs": dict(lr=0.01),
}
scheduler_params = {
    "scheduler": get_cosine_schedule_with_warmup,
    "scheduler_kwargs": dict(num_training_steps=100000, num_warmup_steps=500)
}

# default parameters for GPT
gpt_params = {
    **optim_params, **scheduler_params,
    "config": GPT2Config(
        num_labels=10, hidden_size=256, n_inner=1024,
        n_layer=4, n_head=1, n_positions=100,
    )
}

# default parameters for LSTM
lstm_params = {
    **optim_params, **scheduler_params,
    "input_size": 28*28*1 + 10,
    "hidden_size": 256,
    "batch_first": True
}

# default parameters for MLP
mlp_params = {
    "optimizer": optim.SGD,
    "optimizer_kwargs": dict(lr=0.1),
    "use_batch_norm": False
}

# default transforms for 1-channel images
g_transform = v2.Compose([
    v2.Lambda(lambda x: x.repeat(3, 1, 1, 1)),
    v2.Resize(28),
    v2.Lambda(lambda x: x.transpose(0, 1)),
    v2.Grayscale(),
    v2.Lambda(lambda x: x.flatten(1)),
    v2.ConvertImageDtype(torch.float),
])

# default rgb transform
rgb_transform = [
    v2.Lambda(lambda x: x.transpose(-3, -1)),
    v2.Grayscale(),
    v2.Resize(28),
    v2.Lambda(lambda x: x.flatten(1)),
    v2.ConvertImageDtype(torch.float),
]
cifar_transform = v2.Compose(rgb_transform)
# svhn is different due to transposition
svhn_transform = v2.Compose(rgb_transform[1:])