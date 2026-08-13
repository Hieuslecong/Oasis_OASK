import torch
import torch.nn.functional as F


def soft_erode(x):
    vertical = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
    horizontal = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
    return torch.minimum(vertical, horizontal)


def soft_open(x):
    return F.max_pool2d(soft_erode(x), 3, 1, 1)


def soft_centerline(x, iterations=10):
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")
    x = x.clamp(0, 1)
    center = F.relu(x - soft_open(x))
    for _ in range(int(iterations)):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        center = center + F.relu(delta - center * delta)
    return center.clamp(0, 1)
