import numpy as np


def apply_dp_global(grad_list: list, C: float = 2.0, sigma: float = 0.01) -> list:
    shapes = [g.shape for g in grad_list]
    dtypes = [g.dtype for g in grad_list]

    flat = np.concatenate([g.astype(np.float64).flatten() for g in grad_list])

    norm = np.linalg.norm(flat)
    if norm > C:
        flat = flat * (C / norm)

    noise = np.random.normal(0, sigma * C, size=flat.shape)
    flat_private = flat + noise

    result = []
    offset = 0
    for shape, dtype in zip(shapes, dtypes):
        size = int(np.prod(shape))
        layer_grad = flat_private[offset:offset + size].reshape(shape).astype(dtype)
        result.append(layer_grad)
        offset += size

    return result


def clip_gradients(gradients: np.ndarray, C: float = 1.0) -> np.ndarray:
    norm = np.linalg.norm(gradients)
    if norm > C:
        return gradients * (C / norm)
    return gradients


def add_gaussian_noise(gradients: np.ndarray, C: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    noise = np.random.normal(0, sigma * C, size=gradients.shape)
    return gradients + noise


def apply_dp(gradients: np.ndarray, C: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    p = gradients.astype(np.float32)
    p = clip_gradients(p, C)
    p = add_gaussian_noise(p, C, sigma)
    return p