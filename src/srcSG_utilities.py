import torch
import time

def device_info():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def check_device_and_seed_default(device=None, seed=12345):
    if device is None:
        device = device_info()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return device


def check_device_and_seed(device=None, seed=None):
    if device is None:
        device = device_info()

    # If no seed provided → generate random one
    if seed is None:
        seed = int(time.time() * 1e6) % 2**32

    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    return device

def _to_tensor(x, dtype=None, device=None):
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x, dtype=dtype)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t

def to_tensor(x, dtype=None, device=None):
    t = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype)
    if dtype: t = t.to(dtype)
    if device: t = t.to(device)
    return t