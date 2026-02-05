import gc
from typing import Dict, List, Set, Tuple
import networkit as nk


def create_subgraph(nk_graph: nk.Graph, nodes: Set[int]) -> Tuple[nk.Graph, Dict[int, int]]:
    """
    Create NetworKit subgraph for a given set of nodes with node remapping.
    """
    node_list = sorted(list(nodes))
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(node_list)}
    reverse_mapping = {new_id: old_id for old_id, new_id in node_mapping.items()}

    nk_subgraph = nk.Graph(len(nodes), weighted=True)

    for old_u in nodes:
        for old_v in nk_graph.iterNeighbors(old_u):
            if old_v in nodes and old_u < old_v:  # Avoid duplicate edges
                new_u = node_mapping[old_u]
                new_v = node_mapping[old_v]
                weight = nk_graph.weight(old_u, old_v)
                nk_subgraph.addEdge(new_u, new_v, weight)

    return nk_subgraph, reverse_mapping


def detect_communities(nk_graph: nk.Graph) -> Tuple[List[Set[int]], float]:
    """
    Detect communities in the graph using NetworKit's Louvain method.
    """
    try:
        louvain = nk.community.PLM(nk_graph, refine=True)
        louvain.run()

        partition = louvain.getPartition()
        modularity = nk.community.Modularity().getQuality(partition, nk_graph)

        communities = {}
        for node in range(nk_graph.numberOfNodes()):
            community_id = partition[node]
            if community_id not in communities:
                communities[community_id] = set()
            communities[community_id].add(node)

        return list(communities.values()), modularity

    except Exception as e:
        print(f"NetworKit community detection failed: {e}")
        return [set(range(nk_graph.numberOfNodes()))], 0.0


def recursive_clustering(nk_graph: nk.Graph,
                         cluster_id: str,
                         node_mapping: Dict[int, int],
                         min_cluster_size: int,
                         min_modularity: float,
                         min_graph_size: int) -> Dict[str, List[int]]:
    """
    Recursively apply community detection with sequential subgraph processing.
    """
    print(f"  Cluster {cluster_id}: Recursively clustering subgraph")
    print(f"  Cluster {cluster_id}: Number of nodes: {nk_graph.numberOfNodes()}")
    print(f"  Cluster {cluster_id}: Number of edges: {nk_graph.numberOfEdges()}")

    if node_mapping is None:
        node_mapping = {i: i for i in range(nk_graph.numberOfNodes())}

    original_nodes = [node_mapping[i] for i in range(nk_graph.numberOfNodes())]

    if len(original_nodes) < min_cluster_size:
        print(f"  Cluster {cluster_id}: Stopping recursion - {len(original_nodes)} nodes < {min_cluster_size} (min size)")
        return {cluster_id: original_nodes}

    communities, modularity = detect_communities(nk_graph)

    print(f"  Cluster {cluster_id}: {len(original_nodes)} nodes, modularity = {modularity:.3f}")

    if modularity < min_modularity:
        print(f"  Cluster {cluster_id}: Stopping recursion - modularity {modularity:.3f} < {min_modularity} (min modularity)")
        return {cluster_id: original_nodes}

    if len(communities) <= 1:
        print(f"  Cluster {cluster_id}: Stopping recursion - only {len(communities)} community found")
        return {cluster_id: original_nodes}

    print(f"  Cluster {cluster_id}: Splitting into {len(communities)} subcommunities")
    print(f"  Cluster {cluster_id}: Processing communities sequentially (memory-efficient mode)")

    result = {}
    for i, community in enumerate(communities):
        if len(community) < min_graph_size:
            print(f"  Cluster {cluster_id}.{i + 1}: Ignoring cluster and treating it as noise - {len(community)} nodes < {min_graph_size} (min graph size)")
            continue

        original_community_nodes = {node_mapping[node] for node in community}

        if len(original_community_nodes) < min_cluster_size:
            print(f"  Cluster {cluster_id}.{i + 1}: Stopping recursion and keeping as final cluster - {len(original_community_nodes)} nodes < {min_cluster_size} (min cluster size)")
            sub_cluster_id = f"{cluster_id}.{i + 1}"
            result[sub_cluster_id] = sorted(original_community_nodes)
            continue

        sub_cluster_id = f"{cluster_id}.{i + 1}"

        print(f"  Cluster {sub_cluster_id}: Creating subgraph for {len(community)} nodes")
        nk_subgraph, reverse_mapping = create_subgraph(nk_graph, community)

        composed_mapping = {
            new_id: node_mapping[reverse_mapping[new_id]]
            for new_id in range(nk_subgraph.numberOfNodes())
        }

        try:
            print(f"  Cluster {sub_cluster_id}: Recursively clustering subgraph")
            sub_result = recursive_clustering(
                nk_subgraph,
                sub_cluster_id,
                composed_mapping,
                min_cluster_size,
                min_modularity,
                min_graph_size
            )

            if not sub_result:
                result[sub_cluster_id] = sorted(original_community_nodes)
            else:
                result.update(sub_result)

        finally:
            del nk_subgraph
            del reverse_mapping
            del composed_mapping
            gc.collect()
            print(f"  Cluster {sub_cluster_id}: Subgraph destroyed, memory freed")

    return result
