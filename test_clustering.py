"""Test script for embedding clustering functionality."""
import torch
import numpy as np
from config import Config
import model_utils
from model_clustering import analyze_embeddings

def main():
    """Main function to test clustering analysis."""
    print("Testing embedding clustering analysis...")
    
    # Load configuration
    config = Config()
    
    # Create dataloaders
    _, _, test_loader = model_utils.create_dataloaders(config, remove_fire_labels=False)
    
    print(f"Test loader has {len(test_loader)} batches")
    print(f"Test dataset has {len(test_loader.dataset)} samples")
    
    # Perform embedding analysis
    print("\nStarting embedding analysis...")
    results = analyze_embeddings(config, test_loader)
    
    print("\nClustering analysis completed!")
    print(f"K-Means silhouette score: {results['kmeans_silhouette']:.3f}")
    print(f"DBSCAN silhouette score: {results['dbscan_silhouette']:.3f}")
    print(f"Number of unique K-Means clusters: {len(set(results['kmeans_labels']))}")
    print(f"Number of unique DBSCAN clusters: {len(set(results['dbscan_labels']))}")
    
    # Show cluster distribution
    unique_kmeans, counts_kmeans = np.unique(results['kmeans_labels'], return_counts=True)
    unique_dbscan, counts_dbscan = np.unique(results['dbscan_labels'], return_counts=True)
    
    print("\nK-Means cluster distribution:")
    for cluster, count in zip(unique_kmeans, counts_kmeans):
        print(f"  Cluster {cluster}: {count} samples")
    
    print("\nDBSCAN cluster distribution:")
    for cluster, count in zip(unique_dbscan, counts_dbscan):
        print(f"  Cluster {cluster}: {count} samples")

if __name__ == "__main__":
    main()