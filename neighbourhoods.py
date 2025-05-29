import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans, SpectralClustering
import pandas as pd

def solve_rural_postman_problem_with_partitions(place_name="Speyer, Germany", n_partitions=4, partition_method='geographic'):
    """
    Solve the Rural Postman Problem with neighborhood partitioning.
    
    Parameters:
    - place_name: City name for OSM query
    - n_partitions: Number of neighborhoods to create
    - partition_method: 'geographic', 'spectral', 'community', or 'balanced'
    """
    
    print(f"Downloading street network for {place_name}...")
    start_time = time.time()
    
    # Get the graph from OSM
    G = ox.graph_from_place(place_name, network_type="drive")
    print(f"Graph downloaded in {time.time() - start_time:.2f} seconds")
    print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Convert to undirected graph for RPP
    G_undirected = G.to_undirected()
    
    # Get the largest connected component
    if not nx.is_connected(G_undirected):
        print("Graph is not connected. Using largest connected component.")
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()
        print(f"Largest component has {len(G_undirected.nodes)} nodes and {len(G_undirected.edges)} edges")
    
    # Partition the graph
    partitions = partition_graph(G_undirected, n_partitions, partition_method)
    
    # Solve RPP for each partition
    partition_solutions = solve_partitioned_rpp(G_undirected, partitions, place_name)
    
    # Analyze and visualize results
    analyze_partitioned_solution(G_undirected, partitions, partition_solutions, place_name)
    
    return {
        'original_graph': G_undirected,
        'partitions': partitions,
        'solutions': partition_solutions
    }

def partition_graph(G, n_partitions, method='geographic'):
    """
    Partition the graph into neighborhoods using various methods.
    """
    print(f"\nPartitioning graph into {n_partitions} neighborhoods using {method} method...")
    
    if method == 'geographic':
        return geographic_partition(G, n_partitions)
    elif method == 'spectral':
        return spectral_partition(G, n_partitions)
    elif method == 'community':
        return community_partition(G, n_partitions)
    elif method == 'balanced':
        return balanced_partition(G, n_partitions)
    else:
        raise ValueError(f"Unknown partition method: {method}")

def geographic_partition(G, n_partitions):
    """Partition based on geographic coordinates using K-means clustering."""
    
    if KMeans is None:
        print("Error: scikit-learn not available. Using simple grid-based partitioning instead.")
        return simple_grid_partition(G, n_partitions)
    
    # Extract coordinates
    coords = []
    nodes = []
    for node, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            coords.append([data['x'], data['y']])
            nodes.append(node)
    
    coords = np.array(coords)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_partitions, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    
    # Create partitions
    partitions = {}
    for i, node in enumerate(nodes):
        partition_id = labels[i]
        if partition_id not in partitions:
            partitions[partition_id] = {'nodes': set(), 'edges': set()}
        partitions[partition_id]['nodes'].add(node)
    
    # Add edges that belong to each partition
    for partition_id in partitions:
        partition_nodes = partitions[partition_id]['nodes']
        for u, v in G.edges():
            if u in partition_nodes and v in partition_nodes:
                partitions[partition_id]['edges'].add((u, v))
    
    return partitions

def simple_grid_partition(G, n_partitions):
    """Simple grid-based partitioning when scikit-learn is not available."""
    
    # Get coordinate bounds
    coords = [(data['x'], data['y']) for node, data in G.nodes(data=True) 
              if 'x' in data and 'y' in data]
    
    if not coords:
        # Fallback: random partitioning
        nodes = list(G.nodes())
        partitions = {}
        for i, node in enumerate(nodes):
            partition_id = i % n_partitions
            if partition_id not in partitions:
                partitions[partition_id] = {'nodes': set(), 'edges': set()}
            partitions[partition_id]['nodes'].add(node)
        
        # Add edges
        for partition_id in partitions:
            partition_nodes = partitions[partition_id]['nodes']
            for u, v in G.edges():
                if u in partition_nodes and v in partition_nodes:
                    partitions[partition_id]['edges'].add((u, v))
        
        return partitions
    
    min_x = min(x for x, y in coords)
    max_x = max(x for x, y in coords)
    min_y = min(y for x, y in coords)
    max_y = max(y for x, y in coords)
    
    # Create grid
    grid_size = int(np.ceil(np.sqrt(n_partitions)))
    x_step = (max_x - min_x) / grid_size
    y_step = (max_y - min_y) / grid_size
    
    partitions = {}
    for node, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            x, y = data['x'], data['y']
            grid_x = min(int((x - min_x) / x_step), grid_size - 1)
            grid_y = min(int((y - min_y) / y_step), grid_size - 1)
            partition_id = grid_y * grid_size + grid_x
            
            # Limit to requested number of partitions
            partition_id = partition_id % n_partitions
            
            if partition_id not in partitions:
                partitions[partition_id] = {'nodes': set(), 'edges': set()}
            partitions[partition_id]['nodes'].add(node)
    
    # Add edges
    for partition_id in partitions:
        partition_nodes = partitions[partition_id]['nodes']
        for u, v in G.edges():
            if u in partition_nodes and v in partition_nodes:
                partitions[partition_id]['edges'].add((u, v))
    
    return partitions

def spectral_partition(G, n_partitions):
    """Partition using spectral clustering on the graph Laplacian."""
    
    if SpectralClustering is None:
        print("Warning: scikit-learn not available. Falling back to community detection.")
        return community_partition(G, n_partitions)
    
    # Get adjacency matrix
    nodes = list(G.nodes())
    adj_matrix = nx.adjacency_matrix(G, nodelist=nodes).todense()
    
    # Apply spectral clustering
    spectral = SpectralClustering(n_clusters=n_partitions, random_state=42, 
                                 affinity='precomputed')
    labels = spectral.fit_predict(adj_matrix)
    
    # Create partitions
    partitions = {}
    for i, node in enumerate(nodes):
        partition_id = labels[i]
        if partition_id not in partitions:
            partitions[partition_id] = {'nodes': set(), 'edges': set()}
        partitions[partition_id]['nodes'].add(node)
    
    # Add edges
    for partition_id in partitions:
        partition_nodes = partitions[partition_id]['nodes']
        for u, v in G.edges():
            if u in partition_nodes and v in partition_nodes:
                partitions[partition_id]['edges'].add((u, v))
    
    return partitions

def community_partition(G, n_partitions):
    """Partition using community detection algorithms."""
    
    # Use Louvain community detection
    import networkx.algorithms.community as nx_comm
    
    communities = list(nx_comm.louvain_communities(G, seed=42))
    
    # If we have more communities than requested, merge smallest ones
    while len(communities) > n_partitions:
        # Find two smallest communities and merge them
        sizes = [len(comm) for comm in communities]
        smallest_idx = sizes.index(min(sizes))
        sizes[smallest_idx] = float('inf')  # Exclude from next search
        second_smallest_idx = sizes.index(min(sizes))
        
        # Merge communities
        communities[smallest_idx] = communities[smallest_idx].union(communities[second_smallest_idx])
        communities.pop(second_smallest_idx)
    
    # If we have fewer communities than requested, split largest ones
    while len(communities) < n_partitions:
        # Find largest community and split it
        sizes = [len(comm) for comm in communities]
        largest_idx = sizes.index(max(sizes))
        largest_comm = communities[largest_idx]
        
        if len(largest_comm) > 1:
            # Split randomly
            comm_list = list(largest_comm)
            mid = len(comm_list) // 2
            communities[largest_idx] = set(comm_list[:mid])
            communities.append(set(comm_list[mid:]))
        else:
            break
    
    # Convert to partition format
    partitions = {}
    for i, community in enumerate(communities):
        partitions[i] = {'nodes': community, 'edges': set()}
        for u, v in G.edges():
            if u in community and v in community:
                partitions[i]['edges'].add((u, v))
    
    return partitions

def balanced_partition(G, n_partitions):
    """Create balanced partitions based on edge count."""
    
    # Start with geographic partition
    partitions = geographic_partition(G, n_partitions)
    
    # Balance by moving nodes between partitions
    max_iterations = 10
    for iteration in range(max_iterations):
        # Calculate partition sizes
        partition_sizes = {pid: len(pdata['edges']) for pid, pdata in partitions.items()}
        
        if max(partition_sizes.values()) - min(partition_sizes.values()) < 50:
            break  # Already balanced
        
        # Find most and least loaded partitions
        max_partition = max(partition_sizes, key=partition_sizes.get)
        min_partition = min(partition_sizes, key=partition_sizes.get)
        
        # Move some nodes from max to min partition
        max_nodes = list(partitions[max_partition]['nodes'])
        if len(max_nodes) > 10:  # Only if partition is large enough
            # Move 10% of nodes
            nodes_to_move = max_nodes[:len(max_nodes)//10]
            
            for node in nodes_to_move:
                partitions[max_partition]['nodes'].remove(node)
                partitions[min_partition]['nodes'].add(node)
            
            # Recalculate edges
            for pid in [max_partition, min_partition]:
                partitions[pid]['edges'] = set()
                partition_nodes = partitions[pid]['nodes']
                for u, v in G.edges():
                    if u in partition_nodes and v in partition_nodes:
                        partitions[pid]['edges'].add((u, v))
    
    return partitions

def solve_partitioned_rpp(G, partitions, place_name):
    """Solve RPP for each partition separately."""
    
    solutions = {}
    total_start_time = time.time()
    
    for partition_id, partition_data in partitions.items():
        print(f"\n--- Solving RPP for Partition {partition_id + 1} ---")
        
        # Create subgraph for this partition
        partition_nodes = partition_data['nodes']
        subgraph = G.subgraph(partition_nodes).copy()
        
        if len(subgraph.nodes()) == 0:
            print(f"Partition {partition_id} is empty, skipping...")
            continue
        
        if not nx.is_connected(subgraph):
            print(f"Partition {partition_id} is not connected, using largest component")
            largest_cc = max(nx.connected_components(subgraph), key=len)
            subgraph = subgraph.subgraph(largest_cc).copy()
        
        print(f"Partition {partition_id}: {len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges")
        
        # Solve RPP for this partition
        start_time = time.time()
        solution = solve_rpp_single(subgraph)
        end_time = time.time()
        
        solutions[partition_id] = {
            'subgraph': subgraph,
            'solution': solution,
            'solve_time': end_time - start_time
        }
        
        if solution:
            print(f"Partition {partition_id} solved in {end_time - start_time:.2f} seconds")
            print(f"Tour length: {solution['total_distance']/1000:.2f} km")
        else:
            print(f"Failed to solve partition {partition_id}")
    
    total_time = time.time() - total_start_time
    print(f"\nAll partitions solved in {total_time:.2f} seconds")
    
    return solutions

def solve_rpp_single(G):
    """Solve RPP for a single connected graph component."""
    
    # Find odd-degree vertices
    odd_vertices = [v for v in G.nodes() if G.degree(v) % 2 == 1]
    
    if len(odd_vertices) == 0:
        # Already Eulerian
        eulerian_circuit = list(nx.eulerian_circuit(G))
        return analyze_single_solution(G, eulerian_circuit, [])
    
    if len(odd_vertices) % 2 != 0:
        print("Warning: Odd number of odd-degree vertices - graph may not be properly connected")
        return None
    
    # Find shortest paths between odd vertices
    odd_distances = {}
    for i, u in enumerate(odd_vertices):
        for j, v in enumerate(odd_vertices):
            if i < j:
                try:
                    path_length = nx.shortest_path_length(G, u, v, weight='length')
                    odd_distances[(u, v)] = path_length
                except nx.NetworkXNoPath:
                    odd_distances[(u, v)] = float('inf')
    
    # Find minimum weight perfect matching
    matching_edges = find_min_weight_perfect_matching(odd_vertices, odd_distances)
    
    if not matching_edges:
        return None
    
    # Add matching edges to graph
    G_augmented = G.copy()
    added_edges = []
    
    for u, v in matching_edges:
        try:
            path = nx.shortest_path(G, u, v, weight='length')
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                if G_augmented.has_edge(node1, node2):
                    edge_data = G_augmented[node1][node2]
                    if isinstance(edge_data, dict):
                        valid_data = {k: v for k, v in edge_data.items() if isinstance(k, str)}
                        if 'count' in valid_data:
                            valid_data['count'] += 1
                        else:
                            valid_data['count'] = 2
                        G_augmented.remove_edge(node1, node2)
                        G_augmented.add_edge(node1, node2, **valid_data)
                    else:
                        G_augmented.add_edge(node1, node2, count=2)
                added_edges.append((node1, node2))
        except nx.NetworkXNoPath:
            continue
    
    # Create multigraph and find Eulerian circuit
    G_multi = nx.MultiGraph()
    for u, v, data in G_augmented.edges(data=True):
        count = data.get('count', 1)
        clean_data = {k: v for k, v in data.items() if isinstance(k, str) and k != 'count'}
        for _ in range(count):
            G_multi.add_edge(u, v, **clean_data)
    
    # Check if Eulerian
    odd_vertices_final = [v for v in G_multi.nodes() if G_multi.degree(v) % 2 == 1]
    if len(odd_vertices_final) > 0:
        return None
    
    eulerian_circuit = list(nx.eulerian_circuit(G_multi))
    
    return analyze_single_solution(G, eulerian_circuit, added_edges)

def analyze_single_solution(G, eulerian_circuit, added_edges):
    """Analyze solution for a single partition."""
    
    total_distance = 0
    for u, v in eulerian_circuit:
        if G.has_edge(u, v):
            edge_data = G[u][v]
            if isinstance(edge_data, dict):
                length = edge_data.get('length', 0)
            else:
                length = list(edge_data.values())[0].get('length', 0)
            total_distance += length
    
    original_distance = sum(data.get('length', 0) for u, v, data in G.edges(data=True))
    
    return {
        'tour': eulerian_circuit,
        'added_edges': added_edges,
        'total_distance': total_distance,
        'original_distance': original_distance,
        'efficiency': original_distance/total_distance if total_distance > 0 else 0
    }

def find_min_weight_perfect_matching(vertices, distances):
    """Find minimum weight perfect matching."""
    n = len(vertices)
    if n % 2 != 0:
        return []
    
    if n <= 12:
        return exact_min_weight_perfect_matching(vertices, distances)
    else:
        return greedy_min_weight_perfect_matching(vertices, distances)

def exact_min_weight_perfect_matching(vertices, distances):
    """Exact algorithm for small instances."""
    n = len(vertices)
    min_cost = float('inf')
    best_matching = None
    
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
    
    return best_matching or []

def greedy_min_weight_perfect_matching(vertices, distances):
    """Greedy approximation."""
    remaining = set(vertices)
    matching = []
    
    while len(remaining) > 1:
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
            if len(remaining) >= 2:
                u, v = list(remaining)[:2]
                matching.append((u, v))
                remaining.remove(u)
                remaining.remove(v)
    
    return matching

def analyze_partitioned_solution(G, partitions, solutions, place_name):
    """Analyze and visualize the partitioned RPP solution."""
    
    print(f"\n=== Partitioned RPP Solution Summary for {place_name} ===")
    
    total_distance = 0
    total_original = 0
    valid_solutions = 0
    
    # Create summary table
    summary_data = []
    
    for partition_id, solution_data in solutions.items():
        if solution_data['solution']:
            sol = solution_data['solution']
            total_distance += sol['total_distance']
            total_original += sol['original_distance']
            valid_solutions += 1
            
            summary_data.append({
                'Partition': f"Neighborhood {partition_id + 1}",
                'Nodes': len(solution_data['subgraph'].nodes()),
                'Edges': len(solution_data['subgraph'].edges()),
                'Tour Length (km)': f"{sol['total_distance']/1000:.2f}",
                'Original Length (km)': f"{sol['original_distance']/1000:.2f}",
                'Efficiency (%)': f"{sol['efficiency']*100:.1f}",
                'Solve Time (s)': f"{solution_data['solve_time']:.2f}"
            })
    
    # Print summary table
    df = pd.DataFrame(summary_data)
    print("\nPartition Summary:")
    print(df.to_string(index=False))
    
    print(f"\nOverall Statistics:")
    print(f"Valid partitions solved: {valid_solutions}/{len(partitions)}")
    print(f"Total tour length: {total_distance/1000:.2f} km")
    print(f"Total original length: {total_original/1000:.2f} km")
    print(f"Overall efficiency: {total_original/total_distance*100:.1f}%")
    
    # Visualize partitions
    visualize_partitions(G, partitions, solutions, place_name)

def visualize_partitions(G, partitions, solutions, place_name):
    """Create comprehensive visualization of partitioned solution."""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Partitioned RPP Solution: {place_name}', fontsize=16)
    
    # Get node positions
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    
    # Color map for partitions
    colors = plt.cm.Set3(np.linspace(0, 1, len(partitions)))
    
    # Plot 1: Partition visualization
    ax1 = axes[0, 0]
    for i, (partition_id, partition_data) in enumerate(partitions.items()):
        partition_nodes = list(partition_data['nodes'])
        if partition_nodes:
            subgraph = G.subgraph(partition_nodes)
            nx.draw_networkx_edges(subgraph, pos, edge_color=colors[i], 
                                 width=1, alpha=0.7, ax=ax1)
            nx.draw_networkx_nodes(subgraph, pos, node_color=[colors[i]], 
                                 node_size=10, ax=ax1)
    
    ax1.set_title(f'Graph Partitions ({len(partitions)} neighborhoods)')
    ax1.set_aspect('equal')
    
    # Plot 2: Partition sizes
    ax2 = axes[0, 1]
    partition_sizes = [len(pdata['edges']) for pdata in partitions.values()]
    partition_labels = [f'N{i+1}' for i in range(len(partitions))]
    bars = ax2.bar(partition_labels, partition_sizes, color=colors[:len(partitions)])
    ax2.set_title('Edges per Partition')
    ax2.set_ylabel('Number of Edges')
    
    # Add value labels on bars
    for bar, size in zip(bars, partition_sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(size), ha='center', va='bottom')
    
    # Plot 3: Tour lengths comparison
    ax3 = axes[1, 0]
    valid_partitions = [pid for pid, sol in solutions.items() if sol['solution']]
    if valid_partitions:
        tour_lengths = [solutions[pid]['solution']['total_distance']/1000 
                       for pid in valid_partitions]
        original_lengths = [solutions[pid]['solution']['original_distance']/1000 
                           for pid in valid_partitions]
        labels = [f'N{pid+1}' for pid in valid_partitions]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax3.bar(x - width/2, original_lengths, width, label='Original', alpha=0.8)
        ax3.bar(x + width/2, tour_lengths, width, label='RPP Tour', alpha=0.8)
        
        ax3.set_xlabel('Partition')
        ax3.set_ylabel('Distance (km)')
        ax3.set_title('Tour vs Original Lengths by Partition')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.legend()
    
    # Plot 4: Efficiency comparison
    ax4 = axes[1, 1]
    if valid_partitions:
        efficiencies = [solutions[pid]['solution']['efficiency']*100 
                       for pid in valid_partitions]
        ax4.bar(labels, efficiencies, color='skyblue', alpha=0.8)
        ax4.set_xlabel('Partition')
        ax4.set_ylabel('Efficiency (%)')
        ax4.set_title('Route Efficiency by Partition')
        ax4.set_ylim(0, 100)
        
        # Add value labels
        for i, eff in enumerate(efficiencies):
            ax4.text(i, eff + 1, f'{eff:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Main execution with different partitioning strategies
if __name__ == "__main__":
    # Test different partitioning methods
    methods = ['geographic', 'community', 'balanced']
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"TESTING {method.upper()} PARTITIONING")
        print(f"{'='*60}")
        
        try:
            result = solve_rural_postman_problem_with_partitions(
                "Speyer, Germany", 
                n_partitions=4, 
                partition_method=method
            )
            print(f"{method.capitalize()} partitioning completed successfully!")
        except Exception as e:
            print(f"Error with {method} partitioning: {e}")
            continue