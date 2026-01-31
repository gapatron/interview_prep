"""Solution: get_rank/get_world_size from env; setup_model just to(device); DataLoader stub."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def setup_model_for_distributed(model: nn.Module, device: torch.device) -> nn.Module:
    model = model.to(device)
    # if get_world_size() > 1:
    #     import torch.distributed as dist
    #     model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    return model


def create_dataloader_distributed_ready(
    dataset: Dataset,
    batch_size: int,
    rank: int = None,
    world_size: int = None,
    shuffle: bool = True,
) -> DataLoader:
    if rank is None:
        rank = get_rank()
    if world_size is None:
        world_size = get_world_size()
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
