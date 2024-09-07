import torch
import numpy as np
from torch.utils.data import Sampler

class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, weights, num_samples, num_replicas=None, rank=None, replacement=True):
        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples_per_replica = int(np.ceil(num_samples / num_replicas))
        self.total_size = self.num_samples_per_replica * num_replicas

    def __iter__(self):
        # Perform weighted random sampling
        indices = torch.multinomial(self.weights, self.num_samples, self.replacement)

        # Add extra samples to make it evenly divisible across replicas
        indices = indices.tolist()
        indices += indices[:(self.total_size - len(indices))]
        
        # Subsample for the current replica
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_replica
