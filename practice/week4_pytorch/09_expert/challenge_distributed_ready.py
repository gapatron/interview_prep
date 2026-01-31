"""
Level 5 / Expert — Distributed-Ready Pattern (Single Process)
===============================================================
Structure code so it can be wrapped for DDP: model and data must be prepared
per rank. In single process we use rank 0 only; in multi-GPU we would wrap
model with DistributedDataParallel and use DistributedSampler.
  - get_rank(): return 0 (single process) or int(os.environ["RANK"]) when using torchrun
  - setup_model_for_distributed(model, device, rank): model.to(device); if world_size > 1: wrap with DDP
  - setup_dataloader_for_distributed(dataset, batch_size, rank, world_size): Sampler that partitions data by rank
Implement get_rank(), get_world_size() (return 1 for single process), and a stub
setup_model_for_distributed that just does model.to(device). Optional: add a comment
where DDP and DistributedSampler would go for multi-GPU.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def get_rank() -> int:
    """Return process rank: 0 for single process, or RANK env when using torchrun."""
    # TODO: return int(os.environ.get("RANK", 0))
    pass


def get_world_size() -> int:
    """Return world size: 1 for single process, or WORLD_SIZE env when using torchrun."""
    # TODO: return int(os.environ.get("WORLD_SIZE", 1))
    pass


def setup_model_for_distributed(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Move model to device. In multi-GPU: wrap with DistributedDataParallel.
    For this challenge: just model.to(device) and return model.
    """
    # TODO: model = model.to(device)
    # TODO: if get_world_size() > 1: model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    # TODO: return model
    pass


def create_dataloader_distributed_ready(
    dataset: Dataset,
    batch_size: int,
    rank: int = None,
    world_size: int = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader. For single process: standard shuffle.
    For multi-GPU: use DistributedSampler(dataset, num_replicas=world_size, rank=rank).
    For this challenge: just return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle).
    """
    # TODO: if rank is None: rank = get_rank()
    # TODO: if world_size is None: world_size = get_world_size()
    # TODO: if world_size > 1: sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank); return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    # TODO: return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    pass


if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(4, 2)
    model = setup_model_for_distributed(model, device)
    assert next(model.parameters()).device.type == device.type
    ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
    loader = create_dataloader_distributed_ready(ds, batch_size=4)
    assert len(list(loader)) == 4
    print("Distributed-ready OK.")
