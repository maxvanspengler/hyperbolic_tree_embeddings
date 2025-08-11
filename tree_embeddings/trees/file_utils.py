import json
import os

import networkx as nx


HIERARCHY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)


def load_hierarchy(dataset: str, hierarchy_name: str) -> nx.DiGraph:
    if dataset == "n_h_trees":
        bf, h = [int(arg) for arg in hierarchy_name.split("_")[:2]]
        n_h_tree = nx.balanced_tree(r=bf, h=h, create_using=nx.DiGraph)
        return n_h_tree
    else:
        file = os.path.join(HIERARCHY_DIR, dataset, f"{hierarchy_name}.json")
        with open(file) as f:
            hierarchy_data = json.load(f)
        return nx.node_link_graph(hierarchy_data)


def store_hierarchy(hierarchy: nx.DiGraph, destination: str) -> None:
    json_data = nx.node_link_data(hierarchy)
    with open(destination, "w") as file:
        json.dump(json_data, file, indent=4)


def convert_edges_to_json(dataset: str, filename: str) -> None:
    # Parse paths
    directory = os.path.join(HIERARCHY_DIR, dataset)
    input_file = os.path.join(directory, filename)
    output_file = os.path.join(directory, f"{dataset}.json")

    # Load raw edgelist data and convert to nx-readable format
    with open(input_file, "r") as file_handle:
        raw_edgelist = file_handle.readlines()
    
    # Convert raw edgelist to nx-readable format
    edgelist = []
    for line in raw_edgelist:
        edgelist.append(tuple(int(n) for n in line.split()))

    # Create graph, convert to nodelink format and store as json
    graph = nx.from_edgelist(edgelist)
    json_data = nx.node_link_data(graph)
    with open(output_file, "w") as file_handle:
        json.dump(json_data, file_handle, indent=4)


if __name__ == "__main__":
    import math

    tree = load_hierarchy(dataset="ot_2008", hierarchy_name="ot_08_ot_2008_tree")
    unique_degrees = {d for n, d in nx.degree(tree)}

    # Print some tree properties
    print(tree.number_of_nodes())
    print(len(unique_degrees) - 1)
    print(max(unique_degrees))
    print(nx.diameter(tree.to_undirected(), weight="weight"))
    n = tree.number_of_nodes()
    print(math.ceil(0.5 * (1 + math.sqrt(16 * n - 15))))
