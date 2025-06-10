def create_subgraph(self, nk_graph: nk.Graph, nodes: Set[int]) -> Tuple[nk.Graph, Dict[int, int]]:
        """
        Create NetworKit subgraph for a given set of nodes with node remapping.
        
        Args:
            nk_graph: Original NetworKit graph
            nodes: Set of nodes to include in subgraph
            
        Returns:
            Tuple of (NetworKit subgraph, reverse node mapping dict)
        """
        # Create a mapping from old node IDs to new sequential IDs
        node_list = sorted(list(nodes))
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(node_list)}
        reverse_mapping = {new_id: old_id for old_id, new_id in node_mapping.items()}
        
        # Create new NetworKit graph with remapped nodes
        nk_subgraph = nk.Graph(len(nodes), weighted=True)
        
        # Add edges to the new graph
        for old_u in nodes:
            for old_v in nk_graph.iterNeighbors(old_u):
                if old_v in nodes and old_u < old_v:  # Avoid duplicate edges
                    new_u = node_mapping[old_u]
                    new_v = node_mapping[old_v]
                    weight = nk_graph.weight(old_u, old_v)
                    nk_subgraph.addEdge(new_u, new_v, weight)
        
        return nk_subgraph, reverse_mappingimport numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkit as nk
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

class HierarchicalClustering:
    def __init__(self, similarity_threshold: float = 0.6, min_cluster_size: int = 20, min_modularity: float = 0.5):
        """
        Initialize the hierarchical clustering algorithm.
        
        Args:
            similarity_threshold: Minimum cosine similarity to create an edge (default: 0.6)
            min_cluster_size: Minimum number of nodes in a cluster before stopping recursion (default: 20)
            min_modularity: Minimum modularity score to continue recursion (default: 0.5)
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.min_modularity = min_modularity
        self.final_clusters = {}
        
    def compute_cosine_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity matrix for the embeddings.
        
        Args:
            embeddings: n x d matrix where n is number of items and d is embedding dimension
            
        Returns:
            n x n cosine similarity matrix
        """
        return cosine_similarity(embeddings)
    
    def create_graph_from_embeddings(self, embeddings: np.ndarray) -> nk.Graph:
        """
        Create NetworKit graph from embeddings based on cosine similarity threshold.
        Optimized version without nested loops using vectorized operations.
        
        Args:
            embeddings: n x d matrix of embeddings
            
        Returns:
            NetworKit Graph object
        """
        n_items = embeddings.shape[0]
        similarity_matrix = self.compute_cosine_similarity(embeddings)
        
        # Create NetworKit graph
        nk_graph = nk.Graph(n_items, weighted=True)
        
        # Vectorized edge creation without nested loops
        # Get upper triangular indices (avoid duplicate edges and self-loops)
        i_indices, j_indices = np.triu_indices(n_items, k=1)
        
        # Get similarities for all potential edges
        edge_similarities = similarity_matrix[i_indices, j_indices]
        
        # Find edges that meet the threshold
        valid_edges = edge_similarities > self.similarity_threshold
        
        # Filter to get only valid edge indices and similarities
        valid_i = i_indices[valid_edges]
        valid_j = j_indices[valid_edges]
        valid_similarities = edge_similarities[valid_edges]
        
        # Calculate weights using the new formula: 1 / (1 - cosine_similarity)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        weights = 1.0 / (1.0 - valid_similarities + epsilon)
        
        # Add edges to NetworKit graph
        if len(valid_i) > 0:
            for i, j, weight in zip(valid_i, valid_j, weights):
                nk_graph.addEdge(int(i), int(j), float(weight))
        
        return nk_graph
    
    def detect_communities(self, nk_graph: nk.Graph) -> Tuple[List[Set[int]], float]:
        """
        Detect communities in the graph using NetworKit's Louvain method.
        
        Args:
            nk_graph: NetworKit Graph object
            
        Returns:
            Tuple of (List of sets containing node IDs in communities, modularity score)
        """
        try:
            # Use NetworKit's Louvain algorithm for community detection
            louvain = nk.community.PLM(nk_graph, refine=True)
            louvain.run()
            
            # Get the partition
            partition = louvain.getPartition()
            
            # Calculate modularity score
            modularity = nk.community.Modularity().getQuality(partition, nk_graph)
            
            # Convert partition to communities
            communities = {}
            for node in range(nk_graph.numberOfNodes()):
                community_id = partition[node]
                if community_id not in communities:
                    communities[community_id] = set()
                communities[community_id].add(node)
            
            return list(communities.values()), modularity
            
        except Exception as e:
            print(f"NetworKit community detection failed: {e}")
            # Fallback: return all nodes as a single community with low modularity
            return [set(range(nk_graph.numberOfNodes()))], 0.0
    
    def recursive_clustering(self, nk_graph: nk.Graph, cluster_id: str = "1", 
                           node_mapping: Dict[int, int] = None) -> Dict[str, List[int]]:
        """
        Recursively apply community detection until clusters meet stopping criteria.
        Stops when either:
        1. Number of nodes < min_cluster_size, OR
        2. Modularity score < min_modularity
        
        Args:
            nk_graph: NetworKit Graph object
            cluster_id: Hierarchical ID for the current cluster
            node_mapping: Mapping from current graph node IDs to original node IDs
            
        Returns:
            Dictionary mapping cluster IDs to lists of original node IDs
        """
        # If no mapping provided, create identity mapping
        if node_mapping is None:
            node_mapping = {i: i for i in range(nk_graph.numberOfNodes())}
        
        # Get original node IDs
        original_nodes = [node_mapping[i] for i in range(nk_graph.numberOfNodes())]
        
        # Base case 1: if graph has < min_cluster_size nodes, stop recursion
        if len(original_nodes) < self.min_cluster_size:
            print(f"  Cluster {cluster_id}: Stopping recursion - {len(original_nodes)} nodes < {self.min_cluster_size} (min size)")
            return {cluster_id: original_nodes}
        
        # Detect communities in current graph using NetworKit
        communities, modularity = self.detect_communities(nk_graph)
        
        print(f"  Cluster {cluster_id}: {len(original_nodes)} nodes, modularity = {modularity:.3f}")
        
        # Base case 2: if modularity is below threshold, stop recursion
        if modularity < self.min_modularity:
            print(f"  Cluster {cluster_id}: Stopping recursion - modularity {modularity:.3f} < {self.min_modularity} (min modularity)")
            return {cluster_id: original_nodes}
        
        # Base case 3: if only one community found or community detection failed, stop recursion
        if len(communities) <= 1:
            print(f"  Cluster {cluster_id}: Stopping recursion - only {len(communities)} community found")
            return {cluster_id: original_nodes}
        
        print(f"  Cluster {cluster_id}: Splitting into {len(communities)} subcommunities")
        
        # Recursively process each community
        result = {}
        for i, community in enumerate(communities):
            # Convert community node IDs back to original node IDs
            original_community_nodes = {node_mapping[node] for node in community}
            
            # Create subgraph for this community
            nk_subgraph, reverse_mapping = self.create_subgraph(nk_graph, original_community_nodes)
            
            # Generate hierarchical cluster ID
            sub_cluster_id = f"{cluster_id}.{i + 1}"
            
            # Recursively cluster the subgraph
            sub_result = self.recursive_clustering(nk_subgraph, sub_cluster_id, reverse_mapping)
            result.update(sub_result)
        
        return result
    
    def fit_predict(self, embeddings: np.ndarray) -> Dict[str, List[int]]:
        """
        Main method to perform hierarchical clustering on embeddings.
        
        Args:
            embeddings: n x d matrix of embeddings
            
        Returns:
            Dictionary mapping cluster IDs to lists of node IDs
        """
        print(f"Step 1: Processing {embeddings.shape[0]} items with {embeddings.shape[1]}-dimensional embeddings")
        
        print("Step 2: Creating NetworKit graph from embeddings...")
        nk_graph = self.create_graph_from_embeddings(embeddings)
        print(f"Created graph with {nk_graph.numberOfNodes()} nodes and {nk_graph.numberOfEdges()} edges")
        
        print("Step 3: Applying recursive community detection with NetworKit...")
        clusters = self.recursive_clustering(nk_graph)
        
        print("Step 4: Clustering complete!")
        print(f"Generated {len(clusters)} final clusters")
        
        self.final_clusters = clusters
        return clusters
    
    def print_clusters(self):
        """Print the final clusters in a readable format."""
        print("\n" + "="*50)
        print("FINAL CLUSTERS")
        print("="*50)
        
        for cluster_id, nodes in sorted(self.final_clusters.items()):
            print(f"Cluster {cluster_id}: {len(nodes)} nodes")
            print(f"  Nodes: {sorted(nodes)}")
            print()
        
        print(f"Total clusters: {len(self.final_clusters)}")
        total_nodes = sum(len(nodes) for nodes in self.final_clusters.values())
        print(f"Total nodes: {total_nodes}")

def generate_sample_embeddings(n_items: int = 100, embedding_dim: int = 128, n_clusters: int = 5) -> np.ndarray:
    """
    Generate sample embeddings for testing.
    
    Args:
        n_items: Number of items
        embedding_dim: Dimension of each embedding
        n_clusters: Number of natural clusters to create
        
    Returns:
        n x d matrix of embeddings
    """
    np.random.seed(42)
    embeddings = []
    
    items_per_cluster = n_items // n_clusters
    
    for i in range(n_clusters):
        # Create a cluster center
        center = np.random.randn(embedding_dim)
        
        # Generate embeddings around this center
        for j in range(items_per_cluster):
            # Add some noise to the center
            embedding = center + 0.3 * np.random.randn(embedding_dim)
            # Normalize to unit length for better cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
    
    # Handle remaining items
    remaining = n_items - len(embeddings)
    for i in range(remaining):
        embedding = np.random.randn(embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    
    return np.array(embeddings)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    print("Generating sample embeddings...")
    embeddings = generate_sample_embeddings(n_items=100, embedding_dim=64, n_clusters=4)
    
    # Initialize clustering algorithm with both stopping criteria
    clusterer = HierarchicalClustering(
        similarity_threshold=0.6, 
        min_cluster_size=20, 
        min_modularity=0.5
    )
    
    # Perform clustering
    clusters = clusterer.fit_predict(embeddings)
    
    # Print results
    clusterer.print_clusters()
    
    # Example of how to use your own embeddings:
    """
    # Installation requirements:
    # pip install networkit numpy scikit-learn
    
    # If you have your own embeddings matrix:
    # your_embeddings = np.load('your_embeddings.npy')  # or however you load your data
    # clusterer = HierarchicalClustering(
    #     similarity_threshold=0.6, 
    #     min_cluster_size=20, 
    #     min_modularity=0.5
    # )
    # clusters = clusterer.fit_predict(your_embeddings)
    # clusterer.print_clusters()
    """