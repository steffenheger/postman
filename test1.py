import osmnx as ox
import networkx as nx
from networkx.algorithms import matching, euler

def solve_route_inspection(place_name):
    # Step 1: Load city graph from OpenStreetMap
    print(f"Downloading graph for {place_name}...")
    G = ox.graph_from_place(place_name, network_type='drive')

    # Step 2: Convert to undirected graph
    G = G.to_undirected()

    # Step 3: Keep only the largest connected component
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    # Step 4: Identify all nodes with odd degree
    odd_nodes = [node for node in G.nodes if G.degree(node) % 2 == 1]
    print(f"Found {len(odd_nodes)} odd-degree nodes.")

    # Step 5: Build complete graph of shortest paths between all odd nodes
    print("Computing shortest paths between odd nodes...")
    import itertools
    odd_pairs = list(itertools.combinations(odd_nodes, 2))

    pair_dist = {}
    for u, v in odd_pairs:
        try:
            length = nx.shortest_path_length(G, u, v, weight='length')
            pair_dist[(u, v)] = length
        except nx.NetworkXNoPath:
            continue

    # Step 6: Create a new graph for matching
    matching_graph = nx.Graph()
    for (u, v), dist in pair_dist.items():
        matching_graph.add_edge(u, v, weight=dist)

    # Step 7: Find minimum weight matching of odd nodes
    print("Computing minimum weight matching...")
    min_matching = matching.min_weight_matching(matching_graph)

    # Step 8: Duplicate matched edges' shortest paths to make the graph Eulerian
    print("Duplicating paths to make graph Eulerian...")
    for u, v in min_matching:
        path = nx.shortest_path(G, u, v, weight='length')
        for i in range(len(path) - 1):
            u2, v2 = path[i], path[i + 1]
            if G.has_edge(u2, v2):
                # Duplicate the edge by adding parallel edge with same attributes
                data = G.get_edge_data(u2, v2)[0] if isinstance(G.get_edge_data(u2, v2), dict) else {}
                G.add_edge(u2, v2, **data)

    # Step 9: Graph is now Eulerian — get Eulerian trail
    print("Computing Eulerian trail...")
    trail = list(euler.eulerian_path(G))
    print(f"Trail has {len(trail)} steps (edges).")

    return G, trail


# Example usage
if __name__ == "__main__":
    city = "Speyer, Germany"
    G, trail = solve_route_inspection(city)

    # Optional: Print first few steps
    print("\nFirst few steps of the Eulerian trail:")
    for u, v in trail[:10]:
        print(f"{u} -> {v}")