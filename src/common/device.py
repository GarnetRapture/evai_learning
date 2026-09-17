import torch


def select_torch_device() -> str:
    from common.errors import EvaiError

    if not torch.cuda.is_available():
        raise EvaiError("EVAI training/runtime is fixed to the RTX 3070 CUDA device")
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA GeForce RTX 3070":
        raise EvaiError(f"Expected NVIDIA GeForce RTX 3070, got {name}")
    return "cuda"
