import functools
import math
from typing import Callable

import numpy as np

import torch

from scipy.linalg import hadamard


GOLDEN_RATIO = (1 + 5 ** 0.5) / 2


@functools.cache
def get_sphere_points(
    n: int,
    d: int,
    gen_type: str = "optim",
    obj: str = "min_angles",
) -> torch.Tensor:
    """
    - n:
        Number of points to be generated.
    - d:
        Dimension of hypersphere.
    - gen_type:
        Generation method for obtaining points. Options are:
        - "optim" (default): optimizes points through gradient descent.
        - "hadamard": generate points on inscribed hypercube using Hadamard code.
    - obj:
        Objective function to be used for optimization. Options are:
        - "min_angles" (default): maximizes the minimum angle between points.
        - "min_sims": minimizes the maximum cosine similarity between points.
        - "mhe_log": minimizes the E_0 hyperspherical energy.
        - "mhe_<int>": minimizes the E_s hyperspherical energy for s > 0.
    """
    print(obj, n)
    if d == 2:
        sphere_points = get_2d_sphere_points(n=n)
    elif d == 3:
        sphere_points = get_3d_sphere_points(n=n)
    elif d > 3:
        sphere_points = get_nd_sphere_points(n=n, d=d, gen_type=gen_type, obj=obj)
    else:
        raise ValueError(f"{d} is not a valid embedding dimension.")
    
    return sphere_points


def get_2d_sphere_points(n: int) -> torch.Tensor:
    thetas = (2 * math.pi / n) * torch.arange(n)
    sphere_points = torch.stack(
        tensors=(thetas.cos(), thetas.sin()),
        dim=1,
    )
    return sphere_points


def get_3d_sphere_points(n: int, epsilon: float = 0.3) -> torch.Tensor:
    i = torch.arange(0, n)
    theta = 2 * math.pi * i / GOLDEN_RATIO
    phi = torch.arccos(1 - 2 * (i + epsilon) / (n - 1 + 2 * epsilon))
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)
    return torch.stack((x, y, z), dim=1)


def get_nd_sphere_points(
    n: int, d: int, gen_type: str, obj: str = "min_angles"
) -> torch.Tensor:
    if gen_type == "optim":
        random_points = generate_points_on_hypersphere(n=n, d=d)
        separated_points = optimize_sphere_points(random_points, obj=obj)
    elif gen_type == "hadamard":
        separated_points = hadamard_sphere_points(n=n, d=d)
    return separated_points


def hadamard_sphere_points(n: int, d: int) -> torch.Tensor:
    if n > d:
        raise ValueError(
            f"Can only generate {d} points for {d} dimensions, but {n} were required."
        )

    power_2_d = 2 ** int(math.log2(d))

    hadamard_mat = hadamard(power_2_d)
    hadamard_mat = np.concatenate([hadamard_mat, np.zeros([power_2_d, d - power_2_d])], axis=1)
    sphere_points = hadamard_mat[:n] / math.sqrt(power_2_d)
    print("HADA")
    return torch.tensor(sphere_points)


def generate_points_on_hypersphere(n: int, d: int):
    # Randomly generate spherical coordinates (with r fixed to 1)
    phis = torch.rand(n, d - 1)
    phis = math.pi * phis
    phis[:, -1] = 2 * phis[:, -1]

    # Precompute the sines of the angles phi
    sine_phis = phis.sin()

    # Convert to Cartesian coordinates
    points_on_sphere = torch.empty(n, d)
    for i in range(d):
        points_on_sphere[:, i] = sine_phis[:, :i].prod(dim=1)

        if i < d - 1:
            points_on_sphere[:, i] = phis[:, i].cos() * points_on_sphere[:, i]

    return points_on_sphere


def optimize_sphere_points(
    points: torch.Tensor,
    lr: float = 0.01,
    obj: str = "min_angles",
    max_iter: int = 450,
):
    torch.autograd.set_detect_anomaly(True)
    points = torch.nn.Parameter(points)
    optimizer = torch.optim.SGD([points], lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=150)
    objective_fnc = str_to_obj_map(obj)
    
    i = 1
    while i < max_iter:
        i += 1

        error = objective_fnc(points)
        error.backward()
        points.grad = points.grad.nan_to_num(0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        with torch.no_grad():
            norm_points = points / points.norm(dim=1, keepdim=True)
            points.copy_(norm_points)

    return points


def min_angles_objective(points: torch.Tensor) -> torch.Tensor:
    """
    Computes -\sum_{i = 1}^N \min_{j \neq i}(\arccos(w_i^T w_j))
    """
    pairwise_dots = torch.matmul(points, points.transpose(0, 1))
    angles = torch.arccos(torch.clamp(pairwise_dots, -1 + 1e-6, 1 - 1e-6))
    angles = angles.fill_diagonal_(10)
    min_angles = torch.min(angles, dim=0).values
    return -min_angles.sum()


def min_cosine_similarity(points: torch.Tensor) -> torch.Tensor:
    cosine_similarities = torch.matmul(points, points.transpose(0, 1))
    cosine_similarities = cosine_similarities.fill_diagonal_(-5)
    max_sims = torch.max(cosine_similarities, dim=0).values
    return max_sims.sum()


def mhe_s_objective(points: torch.Tensor, s: int = 1) -> torch.Tensor:
    """
    Computes the MHE objective for s > 0: \sum_{i = 1}^N \sum_{i \neq j} ||w_i - w_j||^{-s}
    """
    pairwise_diff_norms = (points[None, :, :] - points[:, None, :]).norm(dim=-1)
    transformed = 1 / pairwise_diff_norms.pow(s).clamp_min(1e-15)
    transformed = transformed.fill_diagonal_(0.0)
    return transformed.sum()


def mhe_log_objective(points: torch.Tensor) -> torch.Tensor:
    """
    Computes the MHE objective for s = 0: \sum_{i = 1}^N \sum_{i \neq j} \log(||w_i - w_j||^{-1})
    """
    pairwise_diff_norms = (points[None, :, :] - points[:, None, :]).norm(dim=-1)
    transformed = (1 / pairwise_diff_norms.clamp_min(1e-15)).log()
    transformed = transformed.fill_diagonal_(0.0)
    return transformed.sum()


def str_to_obj_map(obj: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Mapping from string to the correct objective function. Options are:
        - min_angles
        - min_sims
        - mhe_log
        - mhe_<int>
    """
    if obj == "min_angles":
        return min_angles_objective
    elif obj == "min_sims":
        return min_cosine_similarity
    elif obj == "mhe_log":
        return mhe_log_objective
    elif "mhe_" in obj:
        s = int(obj.split("_")[-1])
        return functools.partial(mhe_s_objective, s=s)
    else:
        raise KeyError(
            f"Objective function {obj} is unknown"
        )
