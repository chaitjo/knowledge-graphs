import networkx as nx
import matplotlib.pyplot as plt


def draw_kg(triplets):
    k_graph = nx.from_pandas_edgelist(triplets, source='subject', target='object', 
                                      create_using=nx.MultiDiGraph())
    node_deg = nx.degree(k_graph)
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
    labels = dict(zip(
        list(zip(triplets.subject, triplets.object)),
        triplets['relation'].tolist()
    ))
    nx.draw_networkx_edge_labels(
        k_graph, 
        pos=layout, 
        edge_labels=labels,
        font_color='red'
    )
    plt.axis('off')
    plt.show()
    

def draw_kg_subgraph(triplets, node):
    k_graph = nx.from_pandas_edgelist(triplets, source='subject', target='object', 
                                      create_using=nx.MultiDiGraph())
    nodes = [node] + list(nx.dfs_successors(k_graph, node).values())[0]
    print(nodes)
    subgraph = k_graph.subgraph(nodes)
    # layout = nx.spring_layout(subgraph, k=0.15, iterations=20)
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
    labels = dict(zip(
        (list(zip(triplets.subject, triplets.object))),
        triplets['relation'].tolist()
    ))
    edges = tuple(subgraph.out_edges(data=False))
    sublabels = {k: labels[k] for k in edges}
    print(edges)
    print(sublabels)
    nx.draw_networkx_edge_labels(
        subgraph, 
        pos=layout, 
        edge_labels=sublabels,
        font_color='red'
    )
    plt.axis('off')
    plt.show()
