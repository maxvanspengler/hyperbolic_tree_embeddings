# hyperbolic_tree_embeddings
Repository for the ICML 2025 paper "[Low-distorton an GPU-compatible Tree Embeddings in Hyperbolic Space](https://arxiv.org/abs/2502.17130)".

# Requirements & Installation
This repository was built with Python 3.10, so we recommend using `python >= 3.10`. To install the required packages, simply run 
```
pip install -r requirements.txt
```

# Contents
## Tree embeddings
The `tree_embeddings` directory contains the code required for embedding trees with our HS-DTE method or with any of the other methods used in our experiments. The only method not currently included is the constructive approach using precomputed hyperspherical points by Sala et al. (2018), denoted by a ⋆ in the results tables. We intend to add this method to this repository as well, but for now we refer to [their repository](https://github.com/HazyResearch/hyperbolics) for this particular method. Additionally, this repository contains the trees that were used for our experiments and some extra tools for handling graphs or for embeddings graphs into trees.

## Floating point expansions
This is a minimal version of the floating point expansion tensor library that contains the routines required for performing the constructive methods with increased precision. Note that this library was implemented in pure python and does not contain all of PyTorch's usual tensor functionalities. A complete version of this library, along with optimized C++/CUDA implementations is a work in progress. 

# Creating embeddings
To embed a tree using any of the methods contained in this repository, use the `create_embeddings.py` script. This script contains several possible arguments:
- `-d`, `--dataset`: specifies the name of the dataset that the tree originates from. For trees contained in the paper, see the `tree_embeddings/data` directory for the dataset names. See below for additional information on adding new trees.
- `-g`, `--graph-name`: specifies the name (without extension) of the json file containing the actual tree (in NetworkX node-link format).
- `-r`, `--root`: specifies the id of the root node of the tree (default = `0`). If you add your own tree, make sure that you either relabel the nodes such that the root has id 0 or set this argument correctly.
- `-m`, `--method`: specifies whether to use the constructive method (`constructive`), one of the optimization methods (`optimization`) or h-MDS (`h_mds`) (default = `constructive`).
- `-e`, `--embedding-dim`: sets the dimension of the hyperbolic space in which the tree is embedded (default = `20`).
- `-t`, `--tau`: sets the scaling factor tau (default = `1.0`).
- `--terms`: sets the number of terms used for FPEs (default = `1`, so usual floating point arithmetic). This only works for constructive methods. 
- `--dtype`: specifies whether float32's or float64's are used (default: `float64`).
- `--gen-type`: type of hyperspherical generation (`optim` | `hadamard`) that is to be used within the constructive approach (default = `optim`).
- `--optimization-method`: in case of using `-m optimization`, this specifies which of the optimization methods (`distortion` | `hyperbolic_entailment_cones` | `poincare_embeddings`) is used (default = `distortion`).
- `--epochs`: in case of using `-m optimization`, this sets the number of epochs (default = `1000`).
- `--lr`: in case of using `-m optimization`, this sets the learning rate.

Example usages:
```
python create_embeddings.py -d n_h_trees -g 3_5_tree --method constructive --tau 2.0 --terms 2 --gen-type optim
```
```
python create_embeddings.py -d n_h_trees -g 3_5_tree --method optimization --tau 2.0 --optimization-method distortion --epochs 1000 --lr 1.0
```

# Data
All tree data is stored within the `tree_embeddings/data` directory. This directory contains several subdirectories, one for each dataset to which the trees are related. This repository currently contains trees related to datasets:
- cifar100
- cs_phd: computer science PhD advisor-advisee relationships
- diseases: disease relationships
- grqc: general relativity and quantum cosmology arXiv
- ot_601: weevils
- ot_702: lichen
- ot_2008: carnivora
- phylo_tree: mosses

Within each of these dataset directories you can find at least 1 json file containing the node-link format of a NetworkX tree. If you want to add a new tree, you can simply add a json file to one of these directories or, if the tree is related to a different dataset, you can add a new directory containing this json file. To create such a json file from a NetworkX tree, you can either use the `store_hierarchy` function inside the `tree_embeddings/trees/file_utils.py` module or you can directly use NetworkX's [`node_link_data`](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_data.html). 

Aside from these trees, you can also make use of complete m-ary trees, by settings the `dataset` argument to `n_h_trees` and the `graph-name` argument to `<branching_factor>-<depth>-tree`.
