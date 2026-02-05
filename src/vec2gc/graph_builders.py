import numpy as np
import networkit as nk
from typing import List
from joblib import Parallel, delayed

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit length for efficient cosine similarity computation.
    This is done once upfront and reused for all similarity calculations.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)  # Avoid division by zero


def create_graph_from_embeddings_streaming(
    embeddings: np.ndarray,
    similarity_threshold: float,
    batch_size: int = 1000
) -> nk.Graph:
    """
    Create NetworKit graph from embeddings using batched streaming similarity computation.
    """
    n_items = embeddings.shape[0]

    # Normalize embeddings once (reused for all comparisons)
    print(f"Step 2.1: Normalizing embeddings...")
    embeddings_norm = normalize_embeddings(embeddings)
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
            similarities_batch = np.dot(embeddings_norm[batch_start:batch_end], emb_i)

            # Find valid edges in this batch (vectorized filtering)
            valid_mask = similarities_batch > similarity_threshold
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


def create_graph_from_embeddings_streaming_chunked(
    embeddings: np.ndarray,
    similarity_threshold: float,
    chunk_size: int = 1000,
    batch_size: int = 1000,
    n_jobs: int = 1
) -> nk.Graph:
    """
    Memory-efficient streaming version with parallel chunked processing and batched similarity computation.
    """
    n_items = embeddings.shape[0]

    # Normalize embeddings once (shared across all workers)
    print(f"Step 2.1: Normalizing embeddings...")
    embeddings_norm = normalize_embeddings(embeddings)
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
                valid_mask = similarities_batch > similarity_threshold
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


def _bulk_add_edges(nk_graph: nk.Graph, edges_u: List[int], edges_v: List[int], edges_w: List[float]) -> None:
    """
    Bulk-add edges to a NetworKit graph when possible.
    Falls back to per-edge insertion if inputs are empty.
    """
    if not edges_u:
        return
    u_arr = np.asarray(edges_u, dtype=np.int64)
    v_arr = np.asarray(edges_v, dtype=np.int64)
    w_arr = np.asarray(edges_w, dtype=np.float64)
    nk_graph.addEdges(u_arr, v_arr, w_arr)


def create_graph_from_embeddings_faiss_hnsw(
    embeddings: np.ndarray,
    similarity_threshold: float,
    k: int = 30,
    ef_construction: int = 200,
    ef_search: int = 100,
    hnsw_m: int = 16
) -> nk.Graph:
    """
    Create NetworKit graph using FAISS IndexHNSWFlat (approximate k-NN).
    """
    if not _FAISS_AVAILABLE:
        raise ImportError("FAISS is not available. Install faiss or faiss-cpu to use this method.")

    n_items, dim = embeddings.shape
    embeddings_norm = normalize_embeddings(embeddings).astype(np.float32, copy=False)

    index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(embeddings_norm)

    sims, neighbors = index.search(embeddings_norm, k + 1)

    nk_graph = nk.Graph(n_items, weighted=True)
    epsilon = 1e-10
    edges_u: List[int] = []
    edges_v: List[int] = []
    edges_w: List[float] = []

    for i in range(n_items):
        for sim, j in zip(sims[i], neighbors[i]):
            if i == j:
                continue
            if sim < similarity_threshold:
                continue
            if i < j:
                weight = 1.0 / (1.0 - float(sim) + epsilon)
                edges_u.append(int(i))
                edges_v.append(int(j))
                edges_w.append(float(weight))

    _bulk_add_edges(nk_graph, edges_u, edges_v, edges_w)
    return nk_graph


def create_graph_from_embeddings_faiss_flatip(
    embeddings: np.ndarray,
    similarity_threshold: float,
    k: int = 30
) -> nk.Graph:
    """
    Create NetworKit graph using FAISS IndexFlatIP (exact k-NN).
    """
    if not _FAISS_AVAILABLE:
        raise ImportError("FAISS is not available. Install faiss or faiss-cpu to use this method.")

    n_items, dim = embeddings.shape
    embeddings_norm = normalize_embeddings(embeddings).astype(np.float32, copy=False)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_norm)

    sims, neighbors = index.search(embeddings_norm, k + 1)

    nk_graph = nk.Graph(n_items, weighted=True)
    epsilon = 1e-10
    edges_u: List[int] = []
    edges_v: List[int] = []
    edges_w: List[float] = []

    for i in range(n_items):
        for sim, j in zip(sims[i], neighbors[i]):
            if i == j:
                continue
            if sim < similarity_threshold:
                continue
            if i < j:
                weight = 1.0 / (1.0 - float(sim) + epsilon)
                edges_u.append(int(i))
                edges_v.append(int(j))
                edges_w.append(float(weight))

    _bulk_add_edges(nk_graph, edges_u, edges_v, edges_w)
    return nk_graph
