import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, Dataset
from typing import Optional, Union
import numpy as np


def create_sentence_embeddings(
    dataset: Dataset,
    model_name: str,
    text_column: Optional[str] = None,
    split: str = "train",
    batch_size: int = 32,
    device: Optional[str] = None,
    show_progress: bool = True
) -> torch.Tensor:
    """
    Create sentence embeddings for a HuggingFace dataset using sentence-transformers.
    
    Args:
        dataset_name: Name of the HuggingFace dataset (e.g., 'sst2', 'imdb')
        model_name: Name of the sentence-transformer model (e.g., 'all-MiniLM-L6-v2')
        text_column: Name of the text column in the dataset. If None, auto-detects.
        split: Dataset split to use (default: 'train')
        batch_size: Batch size for encoding (default: 32)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        show_progress: Whether to show progress bar (default: True)
        
    Returns:
        torch.Tensor: Embeddings matrix of shape (n_samples, embedding_dim)
    
    Example:
        >>> embeddings = create_sentence_embeddings(
        ...     dataset_name='sst2',
        ...     model_name='all-MiniLM-L6-v2'
        ... )
        >>> print(embeddings.shape)  # (n_samples, 384)
    """
    # Load the model
    print(f"Loading sentence-transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Set device
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    model = model.to(device)
    print(f"Using device: {device}")
    
    # Load the dataset
    # print(f"Loading dataset: {dataset_name} (split: {split})")
    # dataset = load_dataset(dataset_name, split=split)
    
    # Auto-detect text column if not provided
    if text_column is None:
        text_column = _detect_text_column(dataset)
        print(f"Auto-detected text column: {text_column}")
    
    # Extract texts
    texts = dataset[text_column]
    print(f"Processing {len(texts)} sentences...")
    
    # Create embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_tensor=True,
        device=device
    )
    
    print(f"Created embeddings matrix: {embeddings.shape}")
    return embeddings


def _detect_text_column(dataset) -> str:
    """
    Auto-detect the text column in a dataset.
    
    Args:
        dataset: HuggingFace dataset object
        
    Returns:
        str: Name of the text column
    """
    # Common text column names
    text_candidates = ['text', 'sentence', 'content', 'review', 'comment', 'input']
    
    # Check dataset features
    features = dataset.features.keys()
    print(f"Features: {features}")
    
    # Try common names first
    for candidate in text_candidates:
        if candidate in features:
            return candidate
    
    # If no common name found, return the first string column
    for feature_name in features:
        feature_type = dataset.features[feature_name]
        # Check if it's a string type
        if hasattr(feature_type, 'dtype') and 'string' in str(feature_type.dtype).lower():
            return feature_name
        # For datasets library, check if it's Value with string dtype
        if str(feature_type) == 'Value(dtype=string, id=None)':
            return feature_name
    
    # Fallback: return first column
    return list(features)[0]


# Example usage
if __name__ == "__main__":
    # Example 1: Simple usage
    embeddings = create_sentence_embeddings(
        dataset_name='sst2',
        model_name='all-MiniLM-L6-v2',
        split='train'
    )
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Example 2: With custom parameters
    embeddings = create_sentence_embeddings(
        dataset_name='imdb',
        model_name='all-mpnet-base-v2',
        text_column='text',
        batch_size=64,
        show_progress=True
    )
    print(f"Embeddings shape: {embeddings.shape}")