import argparse

import torch

from constructive_tree_embeddings import run_constructive_tree_embeddings
from h_mds_embeddings import run_h_mds_embeddings
from optimization_tree_embeddings import run_optimization_tree_embeddings


parser = argparse.ArgumentParser(
    description="Create embeddings for a given tree."
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    required=True,
    help="Name of the dataset that the tree describes.",
)
parser.add_argument(
    "-g",
    "--graph-name",
    type=str,
    required=True,
    help="Name of the json file containing the tree (without extension).",
)
parser.add_argument(
    "-r",
    "--root",
    type=int,
    default=0,
    help="Root node of the tree (default = 0).",
)
parser.add_argument(
    "-m",
    "--method",
    type=str,
    default="constructive",
    choices=["constructive", "optimization", "h_mds"],
    help="Method to use for embedding the tree (default = 'constructive').",
)
parser.add_argument(
    "-e",
    "--embedding-dim",
    type=int,
    default=20,
    help="Dimension of the embeddings (default = 20).",
)
parser.add_argument(
    "-t",
    "--tau",
    type=float,
    default=1.0,
    help="Hyperbolic radius of the embeddings (default = 1.0).",
)
parser.add_argument(
    "--terms",
    type=int,
    default=1,
    help="Number of terms for floating point expansions (default = 1).",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="float64",
    choices=["float32", "float64"],
    help="Data type for each floating point expansion term (default = 'float64').",
)
parser.add_argument(
    "--gen-type",
    type=str,
    default="optim",
    choices=["optim", "hadamard"],
    help="Type of spherical generation used in constructive method (default = 'optim').",
)
parser.add_argument(
    "--optimization-method",
    type=str,
    default="distortion",
    choices=["distortion", "hyperbolic_entailment_cones", "poincare_embeddings"],
    help="Method to use for optimization-based embeddings (default = 'distortion').",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=1000,
    help="Number of epochs for optimization-based embeddings (default = 200).",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1.0,
    help="Learning rate for optimization-based embeddings (default = 1.0).",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Check for invalid argument combinations
    if args.method != "constructive" and args.terms > 1:
        raise ValueError(
            "Floating point expansions are only supported for constructive embeddings."
        )
    
    # Convert arguments
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    if args.method == "constructive":
        run_constructive_tree_embeddings(
            dataset=args.dataset,
            hierarchy_name=args.graph_name,
            root=args.root,
            gen_type=args.gen_type,
            tau=args.tau,
            embedding_dim=args.embedding_dim,
            nc=args.terms,
            dtype=dtype,
        )
    elif args.method == "optimization":
        run_optimization_tree_embeddings(
            dataset_name=args.dataset,
            hierarchy_name=args.graph_name,
            root=args.root,
            optimization_method=args.optimization_method,
            tau=args.tau,
            embedding_dim=args.embedding_dim,
            dtype=dtype,
            epochs=args.epochs,
            lr=args.lr,
        )
    elif args.method == "h_mds":
        run_h_mds_embeddings(
            dataset=args.dataset,
            hierarchy_name=args.graph_name,
            root=args.root,
            tau=args.tau,
            embedding_dim=args.embedding_dim,
        )
