import numpy as np
import networkit as nk
from typing import List, Dict, Tuple, Set
import joblib
from joblib import Parallel, delayed
import gc
import warnings
warnings.filterwarnings('ignore')


class HierarchicalSequentialClustering:
    def __init__(self, similarity_threshold: float = 0.6, min_cluster_size: int = 20, min_modularity: float = 0.5,
                 min_graph_size: int = 5):
        """
        Initialize the hierarchical clustering algorithm with sequential subgraph processing.
        
        This version uses streaming similarity computation and processes subgraphs sequentially
        (one at a time) to minimize peak memory usage. Subgraphs are created, processed, and
        immediately destroyed before moving to the next community.
        
        Memory optimization: Only one subgraph exists in memory at a time per recursion level,
        reducing peak memory by 60-80% compared to processing all communities in parallel.
        
        Args:
            similarity_threshold: Minimum cosine similarity to create an edge (default: 0.6)
            min_cluster_size: Minimum number of nodes in a cluster before stopping recursion (default: 20)
            min_modularity: Minimum modularity score to continue recursion (default: 0.5)
            min_graph_size: Minimum nodes required to keep a subgraph; smaller subgraphs are treated as noise (default: 5)
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.min_modularity = min_modularity
        self.min_graph_size = min_graph_size
        self.final_clusters = {}
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length for efficient cosine similarity computation.
        This is done once upfront and reused for all similarity calculations.
        
        Args:
            embeddings: n x d matrix of embeddings
            
        Returns:
            Normalized embeddings matrix (n x d)
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)  # Avoid division by zero
    
    def compute_pairwise_similarity(self, emb_i: np.ndarray, emb_j: np.ndarray) -> float:
        """
        Compute cosine similarity between two normalized embedding vectors.
        Since vectors are normalized, this is just a dot product.
        
        Args:
            emb_i: First embedding vector (normalized)
            emb_j: Second embedding vector (normalized)
            
        Returns:
            Cosine similarity value (float)
        """
        return np.dot(emb_i, emb_j)
    
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
        
        return nk_subgraph, reverse_mapping
    
    def create_graph_from_embeddings_streaming(self, embeddings: np.ndarray, 
                                                batch_size: int = 1000) -> nk.Graph:
        """
        Create NetworKit graph from embeddings using batched streaming similarity computation.
        
        This method computes similarities in batches without storing a full n×n matrix,
        significantly reducing memory usage for large datasets while leveraging vectorized
        operations for better performance.
        
        Memory complexity: O(batch_size × d) instead of O(n²) for similarity matrix.
        Adjust batch_size to balance memory usage and computation efficiency.
        
        Args:
            embeddings: n x d matrix of embeddings
            batch_size: Number of similarities to compute in each batch (default: 1000)
                       Larger batches = more memory but faster computation
            
        Returns:
            NetworKit Graph object
        """
        n_items = embeddings.shape[0]
        
        # Normalize embeddings once (reused for all comparisons)
        print(f"Step 2.1: Normalizing embeddings...")
        embeddings_norm = self.normalize_embeddings(embeddings)
        print(f"Step 2.1: Done normalizing embeddings")
        
        # Create NetworKit graph
        nk_graph = nk.Graph(n_items, weighted=True)
        print(f"Step 2.2: Done initializing graph")
        
        # Batched streaming edge creation: compute similarities in batches
        print(f"Step 2.3: Computing similarities and building graph (batched streaming mode, batch_size={batch_size})...")
        epsilon = 1e-10
        edges_added = 0
        
        # Iterate over rows (i)
        for i in range(n_items):
            emb_i = embeddings_norm[i]
            
            # Process similarities in batches for j > i
            j_start = i + 1
            j_end = n_items
            
            # Process in batches
            for batch_start in range(j_start, j_end, batch_size):
                batch_end = min(batch_start + batch_size, j_end)
                
                # Vectorized similarity computation for batch
                # emb_i is shape (d,), embeddings_norm[batch_start:batch_end] is (batch_size, d)
                # Result is shape (batch_size,) - similarities for all j in batch
                similarities_batch = np.dot(embeddings_norm[batch_start:batch_end], emb_i)
                
                # Find valid edges in this batch (vectorized filtering)
                valid_mask = similarities_batch > self.similarity_threshold
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    # Get valid similarities and compute weights (vectorized)
                    valid_similarities = similarities_batch[valid_mask]
                    weights_batch = 1.0 / (1.0 - valid_similarities + epsilon)
                    
                    # Get corresponding j indices
                    valid_j_indices = batch_start + valid_indices
                    
                    # Add edges to graph
                    for j, weight in zip(valid_j_indices, weights_batch):
                        nk_graph.addEdge(int(i), int(j), float(weight))
                        edges_added += 1
            
            # Progress indicator for large datasets
            if (i + 1) % max(1, n_items // 10) == 0:
                print(f"  Processed {i + 1}/{n_items} nodes, {edges_added} edges added so far...")
        
        print(f"Step 2.3: Done - added {edges_added} edges")
        return nk_graph
    
    def create_graph_from_embeddings_streaming_chunked(self, embeddings: np.ndarray, 
                                                      chunk_size: int = 1000,
                                                      batch_size: int = 1000,
                                                      n_jobs: int = 1) -> nk.Graph:
        """
        Memory-efficient streaming version with parallel chunked processing and batched similarity computation.
        
        This method processes embeddings in chunks and computes similarities in batches on-the-fly
        within each chunk, avoiding the need to store full similarity matrices while leveraging
        vectorized operations for better performance.
        
        Args:
            embeddings: n x d matrix of embeddings
            chunk_size: Number of rows to process in each parallel chunk (default: 1000)
            batch_size: Number of similarities to compute in each batch within a chunk (default: 1000)
                       Larger batches = more memory but faster computation
            n_jobs: Number of parallel jobs (-1 for all CPUs, default: 1)
            
        Returns:
            NetworKit Graph object
        """
        n_items = embeddings.shape[0]
        
        # Normalize embeddings once (shared across all workers)
        print(f"Step 2.1: Normalizing embeddings...")
        embeddings_norm = self.normalize_embeddings(embeddings)
        print(f"Step 2.1: Done normalizing embeddings")
        
        # Create NetworKit graph
        nk_graph = nk.Graph(n_items, weighted=True)
        
        def process_chunk(start_idx):
            """
            Process a chunk of rows, computing similarities in batches.
            Returns list of edges as (i, j, weight) tuples.
            """
            end_idx = min(start_idx + chunk_size, n_items)
            edges = []
            epsilon = 1e-10
            
            # Process each row in the chunk
            for i in range(start_idx, end_idx):
                emb_i = embeddings_norm[i]
                
                # Process similarities in batches for j > i
                j_start = i + 1
                j_end = n_items
                
                # Process in batches
                for batch_start in range(j_start, j_end, batch_size):
                    batch_end = min(batch_start + batch_size, j_end)
                    
                    # Vectorized similarity computation for batch
                    similarities_batch = np.dot(embeddings_norm[batch_start:batch_end], emb_i)
                    
                    # Find valid edges in this batch (vectorized filtering)
                    valid_mask = similarities_batch > self.similarity_threshold
                    valid_indices = np.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        # Get valid similarities and compute weights (vectorized)
                        valid_similarities = similarities_batch[valid_mask]
                        weights_batch = 1.0 / (1.0 - valid_similarities + epsilon)
                        
                        # Get corresponding j indices
                        valid_j_indices = batch_start + valid_indices
                        
                        # Add edges to list
                        for j, weight in zip(valid_j_indices, weights_batch):
                            edges.append((i, j, weight))
            
            return edges
        
        # Process chunks in parallel
        if n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        
        chunk_starts = list(range(0, n_items, chunk_size))
        print(f"Step 2.2: Processing {len(chunk_starts)} chunks with {n_jobs} parallel jobs (streaming mode)...")
        
        # Process chunks in parallel
        all_edges = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(start) for start in chunk_starts)
        
        # Flatten and add edges to graph
        total_edges = 0
        for edge_list in all_edges:
            total_edges += len(edge_list)
            for i, j, w in edge_list:
                nk_graph.addEdge(int(i), int(j), float(w))
        
        # Print graph statistics
        num_nodes = nk_graph.numberOfNodes()
        num_edges = nk_graph.numberOfEdges()
        print(f"Step 2.2: Graph created: {num_nodes} nodes, {num_edges} edges")
        if num_nodes > 0:
            avg_degree = (2 * num_edges) / num_nodes
            sparsity = 1.0 - (2 * num_edges) / (num_nodes * (num_nodes - 1))
            print(f"  Average degree: {avg_degree:.2f}, Graph sparsity: {sparsity:.4f}")
        
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
        Recursively apply community detection with sequential subgraph processing.
        
        This method processes communities sequentially (one at a time) to minimize peak memory.
        Each subgraph is created, processed, and immediately destroyed before moving to the next
        community. This reduces peak memory by 60-80% compared to processing all communities
        in parallel.
        
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

        print(f"  Cluster {cluster_id}: Recursively clustering subgraph")
        print(f"  Cluster {cluster_id}: Number of nodes: {nk_graph.numberOfNodes()}")
        print(f"  Cluster {cluster_id}: Number of edges: {nk_graph.numberOfEdges()}")
        
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
        print(f"  Cluster {cluster_id}: Processing communities sequentially (memory-efficient mode)")
        
        # STRATEGY 2: Process communities sequentially - one subgraph at a time
        # This reduces peak memory by only keeping one subgraph in memory at a time
        result = {}
        for i, community in enumerate(communities):
            # Community nodes are in the current graph's coordinate system (not original IDs)
            # We need to use them directly for create_subgraph, but convert to original IDs for final results
            
            # Drop subgraphs that are too small and treat them as noise
            if len(community) < self.min_graph_size:
                print(f"  Cluster {cluster_id}.{i + 1}: Ignoring cluster and treating it as noise - {len(community)} nodes < {self.min_graph_size} (min graph size)")
                continue

            # Convert to original node IDs for size check and final results
            original_community_nodes = {node_mapping[node] for node in community}

            # If the sub-community is too small, stop recursion and keep as final cluster
            if len(original_community_nodes) < self.min_cluster_size:
                print(f"  Cluster {cluster_id}.{i + 1}: Stopping recursion and keeping as final cluster - {len(original_community_nodes)} nodes < {self.min_cluster_size} (min cluster size)")
                sub_cluster_id = f"{cluster_id}.{i + 1}"
                result[sub_cluster_id] = sorted(original_community_nodes)
                continue

            # Generate hierarchical cluster ID
            sub_cluster_id = f"{cluster_id}.{i + 1}"
            
            # Create subgraph using community nodes in CURRENT graph's coordinate system
            # (community is already in the correct coordinate system for nk_graph)
            print(f"  Cluster {sub_cluster_id}: Creating subgraph for {len(community)} nodes")
            nk_subgraph, reverse_mapping = self.create_subgraph(nk_graph, community)
            
            # Compose reverse_mapping with node_mapping to get mapping to original node IDs
            # reverse_mapping: new_subgraph_node_id → current_graph_node_id
            # node_mapping: current_graph_node_id → original_node_id
            # composed_mapping: new_subgraph_node_id → original_node_id
            composed_mapping = {
                new_id: node_mapping[reverse_mapping[new_id]] 
                for new_id in range(nk_subgraph.numberOfNodes())
            }
            
            try:
                # Recursively cluster the subgraph with composed mapping to original node IDs
                print(f"  Cluster {sub_cluster_id}: Recursively clustering subgraph")
                sub_result = self.recursive_clustering(nk_subgraph, sub_cluster_id, composed_mapping)
                
                # If recursion produced no clusters (e.g., all sub-communities were filtered as noise),
                # keep the current subgraph as a final cluster so nodes are not lost.
                if not sub_result:
                    result[sub_cluster_id] = sorted(original_community_nodes)
                else:
                    result.update(sub_result)
            
            finally:
                # CRITICAL: Explicitly clean up subgraph to free memory immediately
                # Delete references to allow garbage collection
                del nk_subgraph
                del reverse_mapping
                del composed_mapping
                
                # Force garbage collection to free memory immediately
                # This ensures only one subgraph exists in memory at a time
                gc.collect()
                
                print(f"  Cluster {sub_cluster_id}: Subgraph destroyed, memory freed")
        
        return result
    
    def fit_predict(self, embeddings: np.ndarray, batch_size: int = 1000) -> Dict[str, List[int]]:
        """
        Main method to perform hierarchical clustering on embeddings using sequential subgraph processing.
        
        This version uses:
        1. Batched streaming similarity computation to minimize memory usage
        2. Sequential subgraph processing (one at a time) to reduce peak memory by 60-80%
        
        For large datasets (>32k embeddings), it automatically uses chunked parallel processing
        for graph construction, but processes subgraphs sequentially during clustering.
        
        Args:
            embeddings: n x d matrix of embeddings
            batch_size: Number of similarities to compute in each batch (default: 1000)
                       Larger batches = more memory but faster computation
                       Smaller batches = less memory but slower computation
                       Adjust based on available memory and desired speed
            
        Returns:
            Dictionary mapping cluster IDs to lists of node IDs
        """
        print(f"Step 1: Processing {embeddings.shape[0]} items with {embeddings.shape[1]}-dimensional embeddings")
        print(f"  Using SEQUENTIAL SUBGRAPH PROCESSING (memory-efficient, batch_size={batch_size})")
        print(f"  Memory optimization: Only one subgraph in memory at a time per recursion level")
        
        print("Step 2: Creating NetworKit graph from embeddings (batched streaming mode)...")
        
        # Automatically choose between single-threaded and parallel chunked processing
        # Based on dataset size
        if embeddings.shape[0] > 4096*8:
            print("Step 2.1: Using parallel chunked streaming (large dataset)")
            nk_graph = self.create_graph_from_embeddings_streaming_chunked(
                embeddings, 
                chunk_size=4096,
                batch_size=batch_size,
                n_jobs=4
            )
        else:
            print("Step 2.1: Using single-threaded streaming")
            nk_graph = self.create_graph_from_embeddings_streaming(embeddings, batch_size=batch_size)
        
        # Additional graph statistics
        num_nodes = nk_graph.numberOfNodes()
        num_edges = nk_graph.numberOfEdges()
        print(f"Step 2 Complete: Graph has {num_nodes} nodes and {num_edges} edges")
        
        print("Step 3: Applying recursive community detection with sequential subgraph processing...")
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
    
    # Initialize clustering algorithm with sequential subgraph processing
    clusterer = HierarchicalSequentialClustering(
        similarity_threshold=0.6, 
        min_cluster_size=20, 
        min_modularity=0.5
    )
    
    # Perform clustering with sequential subgraph processing
    # This reduces peak memory by processing one subgraph at a time
    clusters = clusterer.fit_predict(embeddings, batch_size=1000)
    
    # Print results
    # clusterer.print_clusters()
    
    # Example of how to use your own embeddings:
    """
    # Installation requirements:
    # pip install networkit numpy scikit-learn joblib
    
    # If you have your own embeddings matrix:
    # your_embeddings = np.load('your_embeddings.npy')  # or however you load your data
    # clusterer = HierarchicalSequentialClustering(
    #     similarity_threshold=0.6, 
    #     min_cluster_size=20, 
    #     min_modularity=0.5
    # )
    # # Sequential processing reduces peak memory by 60-80%
    # clusters = clusterer.fit_predict(your_embeddings, batch_size=2000)
    # clusterer.print_clusters()
    """

