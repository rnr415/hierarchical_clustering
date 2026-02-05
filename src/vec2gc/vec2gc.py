import numpy as np
import networkit as nk
from typing import List, Dict, Tuple, Set
import warnings
from vec2gc.graph_builders import (
    normalize_embeddings,
    create_graph_from_embeddings_streaming,
    create_graph_from_embeddings_streaming_chunked,
    create_graph_from_embeddings_faiss_hnsw,
    create_graph_from_embeddings_faiss_flatip,
)
from vec2gc.recursive_clustering import (
    create_subgraph,
    detect_communities,
    recursive_clustering,
)
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
        return normalize_embeddings(embeddings)
    
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
        return create_subgraph(nk_graph, nodes)
    
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
        return create_graph_from_embeddings_streaming(
            embeddings,
            similarity_threshold=self.similarity_threshold,
            batch_size=batch_size
        )
    
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
        return create_graph_from_embeddings_streaming_chunked(
            embeddings,
            similarity_threshold=self.similarity_threshold,
            chunk_size=chunk_size,
            batch_size=batch_size,
            n_jobs=n_jobs
        )

    def create_graph_from_embeddings_faiss_hnsw(self,
                                                embeddings: np.ndarray,
                                                k: int = 30,
                                                ef_construction: int = 200,
                                                ef_search: int = 100,
                                                hnsw_m: int = 16) -> nk.Graph:
        """
        Create NetworKit graph using FAISS IndexHNSWFlat (approximate k-NN).
        """
        return create_graph_from_embeddings_faiss_hnsw(
            embeddings,
            similarity_threshold=self.similarity_threshold,
            k=k,
            ef_construction=ef_construction,
            ef_search=ef_search,
            hnsw_m=hnsw_m
        )

    def create_graph_from_embeddings_faiss_flatip(self,
                                                  embeddings: np.ndarray,
                                                  k: int = 30) -> nk.Graph:
        """
        Create NetworKit graph using FAISS IndexFlatIP (exact k-NN).
        """
        return create_graph_from_embeddings_faiss_flatip(
            embeddings,
            similarity_threshold=self.similarity_threshold,
            k=k
        )

    def detect_communities(self, nk_graph: nk.Graph) -> Tuple[List[Set[int]], float]:
        """
        Detect communities in the graph using NetworKit's Louvain method.
        
        Args:
            nk_graph: NetworKit Graph object
            
        Returns:
            Tuple of (List of sets containing node IDs in communities, modularity score)
        """
        return detect_communities(nk_graph)
    
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
        return recursive_clustering(
            nk_graph=nk_graph,
            cluster_id=cluster_id,
            node_mapping=node_mapping,
            min_cluster_size=self.min_cluster_size,
            min_modularity=self.min_modularity,
            min_graph_size=self.min_graph_size
        )
    
    def fit_predict(self,
                    embeddings: np.ndarray,
                    batch_size: int = 1000,
                    graph_method: str = "auto",
                    k: int = 30,
                    chunk_size: int = 4096,
                    n_jobs: int = 4,
                    ef_construction: int = 200,
                    ef_search: int = 100,
                    hnsw_m: int = 16) -> Dict[str, List[int]]:
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
            graph_method: Graph construction method.
                          Options: "auto", "streaming", "streaming-chunked",
                          "faiss-hnsw", "faiss-flat"
            k: k-NN size for FAISS methods
            chunk_size: Chunk size for streaming-chunked method
            n_jobs: Parallel jobs for streaming-chunked method
            ef_construction: HNSW build parameter (FAISS)
            ef_search: HNSW search parameter (FAISS)
            hnsw_m: HNSW M parameter (FAISS)
            
        Returns:
            Dictionary mapping cluster IDs to lists of node IDs
        """
        print(f"Step 1: Processing {embeddings.shape[0]} items with {embeddings.shape[1]}-dimensional embeddings")
        print(f"  Using SEQUENTIAL SUBGRAPH PROCESSING (memory-efficient, batch_size={batch_size})")
        print(f"  Memory optimization: Only one subgraph in memory at a time per recursion level")
        
        print("Step 2: Creating NetworKit graph from embeddings...")

        if graph_method == "auto":
            # Automatically choose between single-threaded and parallel chunked processing
            # Based on dataset size
            if embeddings.shape[0] > 4096 * 8:
                print("Step 2.1: Using parallel chunked streaming (large dataset)")
                nk_graph = self.create_graph_from_embeddings_streaming_chunked(
                    embeddings,
                    chunk_size=chunk_size,
                    batch_size=batch_size,
                    n_jobs=n_jobs
                )
            else:
                print("Step 2.1: Using single-threaded streaming")
                nk_graph = self.create_graph_from_embeddings_streaming(embeddings, batch_size=batch_size)
        elif graph_method == "streaming":
            print("Step 2.1: Using single-threaded streaming")
            nk_graph = self.create_graph_from_embeddings_streaming(embeddings, batch_size=batch_size)
        elif graph_method == "streaming-chunked":
            print("Step 2.1: Using parallel chunked streaming")
            nk_graph = self.create_graph_from_embeddings_streaming_chunked(
                embeddings,
                chunk_size=chunk_size,
                batch_size=batch_size,
                n_jobs=n_jobs
            )
        elif graph_method == "faiss-hnsw":
            print("Step 2.1: Using FAISS HNSW (approximate k-NN)")
            nk_graph = self.create_graph_from_embeddings_faiss_hnsw(
                embeddings,
                k=k,
                ef_construction=ef_construction,
                ef_search=ef_search,
                hnsw_m=hnsw_m
            )
        elif graph_method == "faiss-flat":
            print("Step 2.1: Using FAISS FlatIP (exact k-NN)")
            nk_graph = self.create_graph_from_embeddings_faiss_flatip(
                embeddings,
                k=k
            )
        else:
            raise ValueError(
                f"Unknown graph_method: {graph_method}. "
                "Use 'auto', 'streaming', 'streaming-chunked', 'faiss-hnsw', or 'faiss-flat'."
            )
        
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
