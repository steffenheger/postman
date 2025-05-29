import osmnx as ox
import networkx as nx
from networkx.algorithms import matching, euler


def getRoute(G):
    print('Step 1: Find odd-degree nodes')
    odd_nodes = [n for n in G.nodes if G.degree(n) % 2 == 1]

    print('Step 2: Build complete graph of shortest paths between odd nodes')
    odd_pairs = [(u, v) for u in odd_nodes for v in odd_nodes if u != v]
    pair_dist = {
        (u, v): nx.shortest_path_length(G, u, v, weight="weight") for u, v in odd_pairs
    }

    print('Step 3: Minimum weight matching')
    odd_graph = nx.Graph()
    odd_graph.add_weighted_edges_from([
        (u, v, dist) for (u, v), dist in pair_dist.items()
    ])

    matching_result = matching.min_weight_matching(odd_graph, maxcardinality=True)

    print('Step 4: Duplicate matched shortest paths in original graph')
    for u, v in matching_result:
        path = nx.shortest_path(G, u, v, weight="weight")
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1], weight=G[path[i]][path[i + 1]]['weight'])

    print('Step 5: Now G is Eulerian – find an Eulerian trail')
    trail = list(euler.eulerian_path(G))  # Not a circuit

    return trail


def getGraph():
    # Download street network for a place
    G = ox.graph_from_place("Speyer, Germany", network_type="drive")

    # Convert to undirected graph for CPP
    G_u = G.to_undirected()

    components = list(nx.connected_components(G_u))
    subgraphs = [G.subgraph(c).copy() for c in components]

    print(len(subgraphs))
    return
    
    if nx.is_eulerian(G_u):
        print('is eulerian')
    else:
        print('is not eulerian')
        trail = getRoute(G)
        print(trail[0])
        print(trail[1])


if __name__ == '__main__':
    getGraph()