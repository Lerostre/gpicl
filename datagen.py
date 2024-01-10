import pandas as pd
import wandb

import numpy as np
import torch
import pytorch_lightning as pl
from torch.nn import functional as F

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from torchvision.transforms import v2
from torchvision.datasets import VisionDataset, MNIST

from models import ClassificationBase, MLP

from tqdm.auto import tqdm
from typing import OrderedDict, Optional
from IPython.display import clear_output

from copy import copy
import os
import random
torch.set_float32_matmul_precision("medium")


def seed_everything(seed):
    """Seed everything you can!"""
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        

class SubLoader:
    """Vision dataset interface for easy transform and indexation"""
    
    def __init__(self, dataset: VisionDataset):
        self.dataset = dataset
        self.data = self.dataset.data
        try:
            self.targets = self.dataset.targets
        except:
            self.targets = self.dataset.labels
        self.targets = torch.tensor(self.targets)
        
    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]
        return data, target
    
    def __repr__(self):
        return self.dataset.__repr__()
    
    def __copy__(self):
        my_copy = type(self)(self.dataset)
        my_copy.__dict__.update(self.__dict__)
        return my_copy
    
    def transform(self, *args, **kwargs):
        return self.dataset.transform(*args, **kwargs)
    
    def __len__(self):
        return len(self.data)


class LinearProjection:
    """
    Linear projector. Has cuda support for faster computation
    and random state for deterministic behaviour.
    Essentially performs M_{dim x dim} @ input_{dim x dim}
    Args:
        dim: data dimensionality for matrix \in R^{dim x dim} creation
        random_state: seed for deterministic generation
        identity: flag for identity tranformation, useful for testing purposes
        device: device for matrix generation, provides substantial speed up
    """
    
    def __init__(self, dim=28*28, random_state=None, identity=False, device="cpu"):
        seed_everything(random_state)
        if device == "cuda":
            self.transformation_matrix = torch.cuda.FloatTensor(dim, dim).normal_(0, 1/dim)
        else:
            self.transformation_matrix = torch.FloatTensor(dim, dim).normal_(0, 1/dim)
        self.identity = identity
        self.device = device
        
    def __call__(self, sample):
        sample = sample.to(self.device)
        res = F.linear(sample, self.transformation_matrix) if not self.identity else sample
        return res.to("cpu")
    

class TargetPermutation:
    """
    Target permuter. Has random state for deterministic behaviour.
    Essentially performs mapping from target vector to a fixed permutation via dict.get
    Args:
        n_targets: number of classes to permute
        random_state: seed for deterministic generation
        identity: flag for identity tranformation, useful for testing purposes
    """
    
    def __init__(self, n_targets=10, random_state=None, identity=False):
        seed_everything(random_state)
        permutation = torch.randperm(n_targets)
        self.mapper = {i: permutation[i].item() for i in range(n_targets)}
        self.identity = identity
            
    def __call__(self, sample):
        return sample.clone().apply_(self.mapper.get) if not self.identity else sample

    
class TaskAugmentor:
    """
    Main class for GPICL data augmentation.
    All transforms are generated on the fly for time and memory efficiency.
    1. Applies all necessary image transforms to whole dataset 
    (might take a while, does not support batched as of yet),
    2. Samples necessary number of objects from n_tasks
    3. Adjusts for sequence sampling if necessary
    4. Projects inputs and permutes labels
    5. Reshapes and concatenates images and labels
    6. Normalizes data along batches and sequences

    Args:
        n_tasks:
            number of tasks to create. Each task corresponds to
            a fixed linear projection and labek permutation of a
            given dataset as a whole
        data_dim: data dimensionality for linear projection
        random_state: seed for deterministic generation
        num_labels: number of classes for target permutation
        device: device for linear projection generation
        draw_sequence:
            behaviour flag for sampling sequences, if True
            implies different sampling logic and makes sure
            that every batch consists of same task objects
        identity: flag for identity tranformation, useful for testing purposes
        sequence_length: length of sequences drawn, ignored if draw_sequence==False

    transform returns:
        batched dataset,
        if draw_sequence == False: ([num_samples x dim], [num_samples])
        else: (
            [num_samples*n_tasks x sequence_length x dim],
            [num_samples*n_tasks x sequence_length]
        )
    """

    def __init__(
        self,
        n_tasks=1,
        data_dim=28*28*1,
        random_state=None,
        num_labels=10,
        device="cuda",
        draw_sequence=False,
        identity=False,
        sequence_length=100,
    ):
        seed_everything(random_state)
        self.data_dim = data_dim
        self.num_labels = 10
        self.draw_sequence = draw_sequence
        self.identity = identity
        self.n_tasks = n_tasks if not self.identity else 1
        self.sequence_length = sequence_length
        self._random_state = random_state
        self._generation_seed = torch.randint(high=int(2**32-1), size=(n_tasks, ))
        self.device = device
        
    def transform(self, dataset=None, n_samples: int = None, normalize=True):
        """
        Transform method itself. Requires dataset, number of samples and normalize flag.
        Sampling is performed per task for sequences and from all data after augment otherwise!
        """
        seed_everything(self._random_state)

        # intricate sampling
        total_size = self.n_tasks*dataset.data.shape[0]
        if self.draw_sequence:
            if n_samples is None:
                n_samples = 1
            # if sequential: sample sequence of sequence_length size n_samples times for each task
            samples = [torch.cat([torch.randint(
                high=dataset.data.shape[0], size=(self.sequence_length,)
            ) for n in range(n_samples)]) for _ in range(self.n_tasks)]
        else:
            if n_samples is not None:
                samples = torch.randint(high=total_size, size=(n_samples,))
            else:
                samples = torch.range(0, total_size)
            # if not sequential: sample from total distribution of len=n_tasks*len(dataset)
            # and split per n_tasks steps for memory efficiency, ensures Uniform distribution
            samples = [samples[
                (samples < total_size*(n+1)/self.n_tasks) &
                (samples >= total_size*n/self.n_tasks)
            ].int() - n*dataset.data.shape[0] for n in range(self.n_tasks)]

        # copy and prepare dataset for populating
        new_dataset = copy(dataset)
        new_dataset.transforms = None
        new_dataset.data = [0]*self.n_tasks
        new_dataset.targets = [0]*self.n_tasks
        data = dataset.transform(dataset.data)
        targets = dataset.targets

        # sample, project, permute for each task
        for n in tqdm(range(self.n_tasks), desc="Generating tasks", leave=False):
            linear_projection = LinearProjection(
                self.data_dim, self._generation_seed[n],
                identity=self.identity, device=self.device
            )
            target_permutation = TargetPermutation(
                self.num_labels, self._generation_seed[n],
                identity=self.identity,
            )
            new_dataset.data[n] = linear_projection(data[samples[n]])
            new_dataset.targets[n] = target_permutation(targets[samples[n]]).unsqueeze(1)

        # reshape depending on draw_sequence
        new_dataset.data = self._to_seq(new_dataset.data, n_samples).to("cpu")
        new_dataset.targets = self._to_seq(new_dataset.targets, n_samples).to("cpu")
        new_dataset.targets = new_dataset.targets.squeeze(-1)
        normalize_dims = [0]
        if self.draw_sequence:
            prev_targets = torch.zeros(self.n_tasks*n_samples, self.sequence_length, 10)
            prev_targets[:, 1:] = F.one_hot(new_dataset.targets[:, :-1], 10)
            new_dataset.data = torch.cat([new_dataset.data, prev_targets], dim=-1)
            normalize_dims.append(1)

        # normalize, don't sleep on it!
        if normalize and n_samples != 1:
            new_dataset.data -= new_dataset.data.mean(dim=tuple(normalize_dims))
            new_dataset.data /= (new_dataset.data.std(dim=tuple(normalize_dims)) + 1e-12)
        
        return new_dataset
    
    def _to_seq(self, output, n_samples):
        """Stacks sequences for sequential sampling if necessary"""
        if self.draw_sequence:
            output = torch.vstack(output)
            return output.reshape(self.n_tasks*n_samples, -1, output.shape[-1])
        else:
            return torch.cat(output)
        
    @property
    def linear_projections(self):
        # projections are not stored for memory efficiency
        return [LinearProjection(
            self.data_dim, self._generation_seed[n], self.identity
        ) for n in range(self.n_tasks)]
    
    @property
    def target_permutations(self):
        # permutations are not stored for memory efficiency
        return [TargetPermutation(
            self.num_labels, self._generation_seed[n], self.identity
        ) for _ in range(self.n_tasks)]