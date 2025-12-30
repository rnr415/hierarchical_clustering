# main.py - Example usage
from embeddings import create_sentence_embeddings
from vec2gc import HierarchicalSequentialClustering
import numpy as np

# Create embeddings

if __name__ == "__main__":
    print("Creating embeddings...")
    embeddings = create_sentence_embeddings(
        dataset_name='sst2',
        model_name='all-MiniLM-L6-v2'
    )

    # Use with clustering
    clusterer = HierarchicalSequentialClustering(
        similarity_threshold=0.6,
        min_cluster_size=20,
        min_modularity=0.5
    )
    clusters = clusterer.fit_predict(embeddings.numpy())