import osmnx as ox
from shapely.geometry import shape
from osmnx import truncate
import json

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

    matching_result = matching.min_weight_matching(odd_graph)

    print('Step 4: Duplicate matched shortest paths in original graph')
    for u, v in matching_result:
        path = nx.shortest_path(G, u, v, weight="weight")
        for i in range(len(path) - 1):
            u_, v_ = path[i], path[i + 1]
            edge_data = G.get_edge_data(u_, v_)

            # If MultiGraph/MultiDiGraph, get first edge
            if isinstance(edge_data, dict) and 0 in edge_data:
                weight = edge_data[0].get("weight", 1)
            elif isinstance(edge_data, dict):
                weight = edge_data.get("weight", 1)
            else:
                weight = 1

            # Add the edge again (duplicates are OK in MultiGraphs)
            G.add_edge(u_, v_, weight=weight)

    print('Step 5: Now G is Eulerian – find an Eulerian trail')
    trail = list(euler.eulerian_path(G))  # Not a circuit

    return trail


def getSubGraph(G, polygon):
    G_clipped = truncate.truncate_graph_polygon(G, polygon)
    return G_clipped

if __name__ == '__main__':
    place_name = "Speyer, Germany"
    G = ox.graph_from_place(place_name, network_type="drive")

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

    # 3. Extract the Shapely polygon
    polygon = shape(geojson['features'][0]['geometry'])

    G_sub = getSubGraph(G, polygon).to_undirected()

    if nx.is_eulerian(G_sub):
        print('is eulerian')
    else:
        print('is not eulerian')
        trail = getRoute(G_sub)
        print(len(trail))
        ordered_nodes = [trail[0][0]] + [v for u, v in trail]
        print(f"Number of edges in trail: {len(trail)}")
        print(f"Number of nodes in path: {len(ordered_nodes)}")

        # Prepare list of nodes with lat/lon and order
        nodes_with_coords = []
        for idx, node_id in enumerate(ordered_nodes, start=1):
            node_data = G_sub.nodes[node_id]
            nodes_with_coords.append({
                "lat": node_data['y'],
                "lon": node_data['x'],
                "order": idx
            })
        print("First 10 nodes in Eulerian path:", nodes_with_coords[:10])
        with open("eulerian_order_points.geojson", "w") as f:
            json.dump(nodes_with_coords, f)
