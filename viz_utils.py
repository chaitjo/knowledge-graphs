import networkx as nx
import matplotlib.pyplot as plt
import itertools


def draw_kg(triplets, save_fig=False):
    # Build networkx graph
    k_graph = nx.from_pandas_edgelist(triplets, source='subject', target='object', 
                                      create_using=nx.MultiDiGraph())
    # Compute node degrees, for resizing highly connected nodes in plot
    node_deg = nx.degree(k_graph)
    # Plot graph
    layout = nx.spring_layout(k_graph, k=0.15, iterations=20)
    plt.figure(num=None, figsize=(120, 90), dpi=80)
    nx.draw_networkx(
        k_graph,
        node_size=[int(deg[1]) * 500 for deg in node_deg],
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color='red',
        edgecolors='black',
        node_color='white',
    )
    # Build edge/relationship labels
    labels = dict(zip(
        list(zip(triplets.subject, triplets.object)),
        triplets['relation'].tolist()
    ))
    # Add edge labels to plot
    nx.draw_networkx_edge_labels(
        k_graph, 
        pos=layout, 
        edge_labels=labels,
        font_color='red'
    )
    plt.axis('off')
    if save_fig:
        plt.savefig("img/kg_full.png", format='png', bbox_inches='tight')
    plt.show()
    

def draw_kg_subgraph(triplets, node, n_hops=2, verbose=True, save_fig=False):
    # Build networkx graph
    k_graph = nx.from_pandas_edgelist(triplets, source='subject', target='object', 
                                      create_using=nx.MultiDiGraph())
    # Build subgraph nodes list
    nodes = [node]
    # Add n-hop DFS successors
    dfs_suc = list(nx.dfs_successors(k_graph, node).values())
    if len(dfs_suc) > 0:
        for hop in range(n_hops):
            nodes += dfs_suc[hop]
    # Build subgraph
    subgraph = k_graph.subgraph(nodes)
    # Plot subgraph
    layout = nx.circular_layout(subgraph)
    plt.figure(num=None, figsize=(10, 10), dpi=80)
    nx.draw_networkx(
        subgraph,
        node_size=1000,
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color='red',
        edgecolors='black',
        node_color='white'
    )
    # Build edge/relationship labels
    labels = dict(zip(
        (list(zip(triplets.subject, triplets.object))),
        triplets['relation'].tolist()
    ))
    edges = tuple(subgraph.out_edges(data=False))
    sublabels = {k: labels[k] for k in edges}
    if verbose:
        for pair in sublabels.keys():
            print("\nS-R-O:\n", pair[0], "-", sublabels[pair], "-", pair[1])
    # Add edge labels to plot
    nx.draw_networkx_edge_labels(
        subgraph, 
        pos=layout, 
        edge_labels=sublabels,
        font_color='red'
    )
    plt.axis('off')
    if save_fig:
        plt.savefig(f"img/kg_{node.lower()}.png", format='png', bbox_inches='tight')
    plt.show()
