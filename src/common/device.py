import torch


def select_torch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
