import osmnx as ox
import networkx as nx
#import numpy as np
#from itertools import combinations
import matplotlib.pyplot as plt
#from collections import defaultdict
import time
from osmnx import truncate
from shapely.geometry import shape

def getSubGraph(G, polygon):
    G_clipped = truncate.truncate_graph_polygon(G, polygon)
    return G_clipped

def solve_rural_postman_problem(G, place_name="Speyer, Germany"):
    """
    Solve the Rural Postman Problem on a city's street network.
    
    The Rural Postman Problem finds the shortest closed walk that visits
    every edge at least once in a graph that may not be Eulerian.
    """
    
    # Convert to undirected graph for RPP
    G_undirected = G.to_undirected()
    
    # Get the largest connected component
    if not nx.is_connected(G_undirected):
        print("Graph is not connected. Using largest connected component.")
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()
        print(f"Largest component has {len(G_undirected.nodes)} nodes and {len(G_undirected.edges)} edges")
    
    return solve_rpp(G_undirected, place_name)

def solve_rpp(G, place_name):
    """
    Solve Rural Postman Problem using the standard algorithm:
    1. Find odd-degree vertices
    2. Find minimum weight perfect matching on odd vertices
    3. Add matching edges to make graph Eulerian
    4. Find Eulerian circuit
    """
    
    print("\nSolving Rural Postman Problem...")
    
    # Step 1: Check if graph is already Eulerian
    odd_vertices = [v for v in G.nodes() if G.degree(v) % 2 == 1]
    print(f"Found {len(odd_vertices)} odd-degree vertices")
    
    if len(odd_vertices) == 0:
        print("Graph is already Eulerian!")
        eulerian_circuit = list(nx.eulerian_circuit(G))
        return analyze_solution(G, eulerian_circuit, [], place_name)
    
    # Step 2: Find shortest paths between all pairs of odd vertices
    print("Computing shortest paths between odd vertices...")
    odd_distances = {}
    
    for i, u in enumerate(odd_vertices):
        for j, v in enumerate(odd_vertices):
            if i < j:  # Only compute once for each pair
                try:
                    path_length = nx.shortest_path_length(G, u, v, weight='length')
                    odd_distances[(u, v)] = path_length
                except nx.NetworkXNoPath:
                    print(f"No path between {u} and {v}")
                    odd_distances[(u, v)] = float('inf')
    
    # Step 3: Find minimum weight perfect matching
    print("Finding minimum weight perfect matching...")
    matching_edges = find_min_weight_perfect_matching(odd_vertices, odd_distances)
    
    # Step 4: Add matching edges to graph
    G_augmented = G.copy()
    added_edges = []
    
    for u, v in matching_edges:
        # Get the actual path and add all edges in the path
        try:
            path = nx.shortest_path(G, u, v, weight='length')
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                if G_augmented.has_edge(node1, node2):
                    # Add parallel edge by incrementing edge count
                    edge_data = G_augmented[node1][node2]
                    if isinstance(edge_data, dict):
                        # Filter out non-string keys and copy valid attributes
                        valid_data = {k: v for k, v in edge_data.items() if isinstance(k, str)}
                        if 'count' in valid_data:
                            valid_data['count'] += 1
                        else:
                            valid_data['count'] = 2
                        # Remove old edge and add new one with updated count
                        G_augmented.remove_edge(node1, node2)
                        G_augmented.add_edge(node1, node2, **valid_data)
                    else:
                        # Handle MultiGraph case
                        G_augmented.add_edge(node1, node2, count=2)
                added_edges.append((node1, node2))
        except nx.NetworkXNoPath:
            print(f"Warning: No path found for matching edge ({u}, {v})")
    
    print(f"Added {len(added_edges)} edge traversals to make graph Eulerian")
    
    # Step 5: Find Eulerian circuit
    print("Finding Eulerian circuit...")
    
    # Create multigraph to handle multiple edges
    G_multi = nx.MultiGraph()
    for u, v, data in G_augmented.edges(data=True):
        count = data.get('count', 1)
        # Only keep string keys for edge attributes
        clean_data = {k: v for k, v in data.items() if isinstance(k, str) and k != 'count'}
        for _ in range(count):
            G_multi.add_edge(u, v, **clean_data)
    
    # Verify the graph is now Eulerian
    odd_vertices_final = [v for v in G_multi.nodes() if G_multi.degree(v) % 2 == 1]
    if len(odd_vertices_final) > 0:
        print(f"Warning: Graph still has {len(odd_vertices_final)} odd vertices")
        return None
    
    # Find Eulerian circuit
    eulerian_circuit = list(nx.eulerian_circuit(G_multi))
    
    return analyze_solution(G, eulerian_circuit, added_edges, place_name)

def find_min_weight_perfect_matching(vertices, distances):
    """
    Find minimum weight perfect matching using brute force for small sets,
    or a greedy approximation for larger sets.
    """
    n = len(vertices)
    if n % 2 != 0:
        raise ValueError("Cannot find perfect matching on odd number of vertices")
    
    if n <= 12:  # Use exact algorithm for small instances
        return exact_min_weight_perfect_matching(vertices, distances)
    else:  # Use greedy approximation for larger instances
        return greedy_min_weight_perfect_matching(vertices, distances)

def exact_min_weight_perfect_matching(vertices, distances):
    """Exact algorithm using brute force enumeration."""
    n = len(vertices)
    min_cost = float('inf')
    best_matching = None
    
    # Generate all possible perfect matchings
    def generate_matchings(remaining_vertices):
        if len(remaining_vertices) == 0:
            return [[]]
        if len(remaining_vertices) == 2:
            return [[(remaining_vertices[0], remaining_vertices[1])]]
        
        matchings = []
        first = remaining_vertices[0]
        for i in range(1, len(remaining_vertices)):
            partner = remaining_vertices[i]
            rest = remaining_vertices[1:i] + remaining_vertices[i+1:]
            for sub_matching in generate_matchings(rest):
                matchings.append([(first, partner)] + sub_matching)
        return matchings
    
    all_matchings = generate_matchings(vertices)
    
    for matching in all_matchings:
        cost = sum(distances.get((min(u, v), max(u, v)), float('inf')) for u, v in matching)
        if cost < min_cost:
            min_cost = cost
            best_matching = matching
    
    return best_matching

def greedy_min_weight_perfect_matching(vertices, distances):
    """Greedy approximation algorithm."""
    remaining = set(vertices)
    matching = []
    
    while len(remaining) > 0:
        # Find the minimum weight edge among remaining vertices
        min_weight = float('inf')
        best_edge = None
        
        for u in remaining:
            for v in remaining:
                if u != v:
                    key = (min(u, v), max(u, v))
                    if key in distances and distances[key] < min_weight:
                        min_weight = distances[key]
                        best_edge = (u, v)
        
        if best_edge:
            matching.append(best_edge)
            remaining.remove(best_edge[0])
            remaining.remove(best_edge[1])
        else:
            # Fallback: pair first two remaining vertices
            if len(remaining) >= 2:
                u, v = list(remaining)[:2]
                matching.append((u, v))
                remaining.remove(u)
                remaining.remove(v)
    
    return matching

def analyze_solution(G, eulerian_circuit, added_edges, place_name):
    """Analyze and visualize the RPP solution."""
    
    # Calculate total distance
    total_distance = 0
    original_distance = 0
    
    for u, v in eulerian_circuit:
        if G.has_edge(u, v):
            edge_data = G[u][v]
            if isinstance(edge_data, dict):
                length = edge_data.get('length', 0)
            else:  # MultiGraph case
                length = list(edge_data.values())[0].get('length', 0)
            total_distance += length
    
    # Calculate original graph total length
    for u, v, data in G.edges(data=True):
        original_distance += data.get('length', 0)
    
    # Calculate added distance more carefully
    added_distance = 0
    for u, v in added_edges:
        if G.has_edge(u, v):
            edge_data = G[u][v]
            if isinstance(edge_data, dict):
                added_distance += edge_data.get('length', 0)
            else:
                # Handle MultiGraph case - get first edge data
                added_distance += list(edge_data.values())[0].get('length', 0)
    
    print(f"\n=== Rural Postman Problem Solution for {place_name} ===")
    print(f"Original graph total length: {original_distance/1000:.2f} km")
    print(f"Added edges total length: {added_distance/1000:.2f} km")
    print(f"RPP tour total length: {total_distance/1000:.2f} km")
    print(f"Tour has {len(eulerian_circuit)} edge traversals")
    print(f"Efficiency: {original_distance/total_distance*100:.1f}% (lower is more backtracking)")
    
    # Visualize the solution
    visualize_solution(G, eulerian_circuit, added_edges, place_name)
    
    return {
        'graph': G,
        'tour': eulerian_circuit,
        'added_edges': added_edges,
        'total_distance': total_distance,
        'original_distance': original_distance,
        'efficiency': original_distance/total_distance
    }

def visualize_solution(G, eulerian_circuit, added_edges, place_name):
    """Create visualization of the RPP solution."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Original graph with added edges highlighted
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    
    # Draw original edges
    ox.plot_graph(G, ax=ax1, node_size=0, edge_color='lightblue', 
                  edge_linewidth=0.5, show=False, close=False)
    
    # Highlight added edges
    if added_edges:
        added_edge_list = [(u, v) for u, v in added_edges if G.has_edge(u, v)]
        if added_edge_list:
            nx.draw_networkx_edges(G, pos, edgelist=added_edge_list, 
                                 edge_color='red', width=2, ax=ax1)
    
    ax1.set_title(f'RPP Solution: Original Graph + Added Edges\n{place_name}')
    ax1.legend(['Original edges', 'Added edges for Eulerian circuit'])
    
    # Plot 2: Tour visualization (sample of edges to avoid overcrowding)
    ox.plot_graph(G, ax=ax2, node_size=0, edge_color='lightgray', 
                  edge_linewidth=0.3, show=False, close=False)
    
    # Draw a sample of the tour (every 10th edge to avoid overcrowding)
    sample_circuit = eulerian_circuit[::max(1, len(eulerian_circuit)//100)]
    tour_edges = [(u, v) for u, v in sample_circuit if G.has_edge(u, v)]
    
    if tour_edges:
        nx.draw_networkx_edges(G, pos, edgelist=tour_edges, 
                             edge_color='orange', width=1, ax=ax2, alpha=0.7)
    
    ax2.set_title(f'Sample of Eulerian Tour\n(showing ~{len(sample_circuit)} of {len(eulerian_circuit)} edges)')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Main execution
if __name__ == "__main__":
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "coordinates": [[
                        [8.421755809732616, 49.35124935839016],
                        [8.419390347008829, 49.3478846880686],
                        [8.428287959092387, 49.34074460440627],
                        [8.43256315318996, 49.340673905328686],
                        [8.442046705581703, 49.34775744812055],
                        [8.437706407004413, 49.35321433251107],
                        [8.425813988902433, 49.35293160725601],
                        [8.421755809732616, 49.35124935839016]
                    ]],
                    "type": "Polygon"
                }
            }
        ]
    }

    polygon = shape(geojson['features'][0]['geometry'])

    # Get the graph from OSM
    G = ox.graph_from_place('Speyer, Germany', network_type="drive")
    G_sub = getSubGraph(G, polygon)

    # Solve RPP for Speyer, Germany
    solution = solve_rural_postman_problem(G_sub)
    
    if solution:
        print(f"\nRural Postman Problem solved successfully!")
        print(f"The mail carrier would need to travel {solution['total_distance']/1000:.2f} km")
        print(f"to traverse every street at least once and return to the starting point.")
    else:
        print("Failed to solve Rural Postman Problem")