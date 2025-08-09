import networkx
import numpy as np
def partition(embeddings):
    
    dist = np.dot(embeddings)
def kruskal(adj):
    MST = set()
    edges = set()
    num_nodes = adj.shape[0]
    for j in range(num_nodes):
        for k in range(num_nodes):
            if G.graph[j][k] != 0 and (k, j) not in edges:
                edges.add((j, k))
    sorted_edges = sorted(edges, key=lambda e:G.graph[e[0]][e[1]])
    uf = UF(G.vertices)
    for e in sorted_edges:
        u, v = e
        if uf.connected(u, v):
            continue
        uf.union(u, v)
        MST.add(e)
    return MST