#!/usr/bin/env python3
"""Simple runnable example for vec2gc

Usage:
  python examples/run_clustering.py --mode synthetic
  python examples/run_clustering.py --mode dataset  # uses a small HF dataset if available

This script is deliberately lightweight and aims to be runnable in CI or locally.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

try:
    # Local import from package
    from vec2gc.vec2gc import HierarchicalSequentialClustering, generate_sample_embeddings
except Exception as e:  # pragma: no cover - example script
    raise


def run_synthetic(n_items: int = 200, dim: int = 128, n_clusters: int = 4, batch_size: int = 1000):
    print("Generating synthetic embeddings...")
    emb = generate_sample_embeddings(n_items=n_items, embedding_dim=dim, n_clusters=n_clusters)

    clusterer = HierarchicalSequentialClustering(
        similarity_threshold=0.6, min_cluster_size=10, min_modularity=0.5
    )

    clusters = clusterer.fit_predict(emb, batch_size=batch_size)

    print_summary(clusters)
    return clusters


def run_dataset(sample_size: int = 500, batch_size: int = 512):
    # Optional: use HuggingFace dataset & sentence-transformers (may be slow if not cached)
    try:
        from datasets import load_dataset
        from vec2gc.embeddings import create_sentence_embeddings
    except Exception as e:  # pragma: no cover - optional demo
        raise RuntimeError("Missing optional packages for dataset mode: install sentence-transformers and datasets") from e

    print("Loading small dataset subset (AG News)...")
    ds = load_dataset("ag_news", split=f"train[:{sample_size}]")

    emb = create_sentence_embeddings(ds, model_name="all-MiniLM-L6-v2", text_column="text", batch_size=64)
    emb_np = emb.numpy()

    clusterer = HierarchicalSequentialClustering(similarity_threshold=0.6, min_cluster_size=20)
    clusters = clusterer.fit_predict(emb_np, batch_size=batch_size)

    print_summary(clusters)
    return clusters


def print_summary(clusters: dict):
    n_clusters = len(clusters)
    sizes = sorted([len(v) for v in clusters.values()], reverse=True)
    total_nodes = sum(sizes)

    print("\nRESULT SUMMARY")
    print("--------------")
    print(f"Clusters: {n_clusters}")
    print(f"Top cluster sizes: {sizes[:5]}")
    print(f"Total nodes assigned: {total_nodes}")

    # Save to disk for quick inspection
    out_path = os.path.join(".", "clusters.json")
    with open(out_path, "w") as fh:
        json.dump(clusters, fh)
    print(f"Saved clusters.json ({out_path})")


def main(mode: str, **kwargs):
    if mode == "synthetic":
        return run_synthetic(**kwargs)
    elif mode == "dataset":
        return run_dataset(**kwargs)
    else:
        raise ValueError("Unknown mode: " + str(mode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "dataset"], default="synthetic")
    parser.add_argument("--n_items", type=int, default=200)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    main(args.mode, n_items=args.n_items, dim=args.dim, n_clusters=args.n_clusters, batch_size=args.batch_size)
