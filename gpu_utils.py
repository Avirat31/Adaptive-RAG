from __future__ import annotations


def get_gpu_ids(num_gpus: int | None = None) -> list[int]:
    """Visible CUDA device ids (respects CUDA_VISIBLE_DEVICES). Does not create a CUDA context."""
    try:
        import torch
        total = torch.cuda.device_count()
    except Exception:
        total = 0
    ids = list(range(total))
    return ids[:num_gpus] if num_gpus else ids


def free_cuda():
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
