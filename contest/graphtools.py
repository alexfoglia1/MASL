import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_graph(adjmatrix,labels):
    adjacency_matrix = np.matrix(adjmatrix)
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr = gr.to_directed()
    for i in range(0,len(adjmatrix)):
        gr.add_node(i)
    gr.add_edges_from(edges)
    if labels is None:
        labels = range(len(gr))
    lbls = dict(zip(gr,labels))
    nx.draw(gr, node_size=500, labels=lbls, with_labels=True)
    plt.show()
