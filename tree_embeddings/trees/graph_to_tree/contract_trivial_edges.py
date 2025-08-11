import networkx as nx

from tqdm import tqdm


def get_minimal_edge_weigth(t: nx.Graph):
    edge_weights = [e[2]["weight"] for e in list(t.edges(data=True))]
    return min(edge_weights)


def contract_tree(tree: nx.Graph, n: int) -> nx.Graph:
    min_weight = get_minimal_edge_weigth(tree)
    while min_weight == 0:
        edges = list(tree.edges())
        for e in tqdm(edges):
            # Skip if source or target has already been contracted
            if e[0] not in tree.nodes() or e[1] not in tree.nodes():
                continue

            # If an edge weight is 0, contract into the node with smaller id
            weight = tree.get_edge_data(*e)["weight"]
            if weight == 0:
                if e[1] < e[0]:
                    e = (e[1], e[0])

                # If both nodes of the edge are in the original graph, throw an error
                if e[1] < n:
                    raise RuntimeError(
                        f"There's a zero weighted edge between 2 of the original graph nodes. "
                        f"This would cause one of the original nodes to disappear, which cannot "
                        f"happen."
                    )
                
                nx.contracted_edge(tree, e, self_loops=False, copy=False)
        
        min_weight = get_minimal_edge_weigth(tree)

    return tree
