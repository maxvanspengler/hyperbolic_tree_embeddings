import math
import os

import networkx as nx

import torch

import matplotlib.pyplot as plt


def plot_embeddings(
    hierarchy_embeddings: torch.Tensor,
    hierarchy: nx.DiGraph,
    tau: float,
    fig_dir: str,
    n: int = 1000,
) -> None:
    if hierarchy_embeddings.size(1) != 2:
        raise ValueError(
            f"Visualization only works for embedding dimension 2, but got "
            f"{hierarchy_embeddings.size(1)}"
        )
    
    n_edges = hierarchy.number_of_edges()
    
    # Initialize tensors x and y
    x = torch.empty(n_edges, 2)
    y = torch.empty(n_edges, 2)

    # For each edge add the parent embedding to x and the child embedding to y
    for idx, (parent, child) in enumerate(hierarchy.edges):
        x[idx] = hierarchy_embeddings[parent]
        y[idx] = hierarchy_embeddings[child]

    # Compute the centers and radii of the circles that contain the x -> y arcs
    triangle_sizes = x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
    c = (
        (x + y) / 2
        + (
            (1 - torch.einsum("bd,bd->b", x, y))
            / (2 * triangle_sizes)
        )[:, None] * torch.stack(
            (y[:, 1] - x[:, 1], x[:, 0] - y[:, 0]),
            dim=1,
        )
    )
    r = (c - x).norm(dim=1)

    # Compute the thetas that parametrize the x -> y arcs
    thetax = 2 * ((x[:, 1] - c[:, 1]) / (x[:, 0] - c[:, 0] + r)).arctan() % (2 * math.pi)
    thetay = 2 * ((y[:, 1] - c[:, 1]) / (y[:, 0] - c[:, 0] + r)).arctan() % (2 * math.pi)

    # Need to make sure we take the arc inside the Poincaré disk
    theta_diff = (thetay - thetax).abs()
    theta_diff = torch.where(theta_diff < math.pi, theta_diff, theta_diff - 2 * math.pi)

    # And need to make sure that we move in the right direction
    theta_min = torch.where(thetax < thetay, thetax, thetay)

    # Create a mesh over the arcs
    theta = theta_min[:, None] + theta_diff[:, None] / n * torch.arange(0, n)
    arc = c[:, :, None] + r[:, None, None] * torch.stack(
        [theta.cos(), theta.sin()],
        dim=1,
    )

    # Correct for case where x and y lie on circle diameter
    t = 1 / n * torch.arange(0, n)
    straight_lines = x[:, :, None] * t + y[:, :, None] * (1 - t)
    cond = triangle_sizes.isclose(torch.tensor(0.0), atol=1e-6)
    arc = torch.where(cond[:, None, None], straight_lines, arc)

    # Plot the arcs and the embedded points
    _, ax = plt.subplots()
    for p in range(n_edges):
        ax.plot(arc.numpy()[p, 0, :], arc.numpy()[p, 1, :], c="black", linewidth=1, zorder=1)
    ax.scatter(x=x.numpy()[:, 0], y=x.numpy()[:, 1], s=25, c="tab:blue", zorder=2)
    ax.scatter(x=y.numpy()[:, 0], y=y.numpy()[:, 1], s=25, c="tab:blue", zorder=2)
    ax.add_patch(
        plt.Circle((0, 0), 1, color="black", fill=False)
    )
    ax.set_aspect("equal")

    ax.set_axis_off()

    # Save figure
    print(fig_dir)
    print(f"hierarchy_embedding_tau_{tau}.png")
    plt.savefig(
        os.path.join(
            fig_dir, f"hierarchy_embedding_tau_{tau}.png"
        ),
        dpi=300,
    )
    plt.close()
