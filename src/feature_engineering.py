"""
Feature Engineering Module for Cyber Security Threat Detection.
Performs feature selection, dimensionality reduction, and importance ranking.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def remove_low_variance_features(X, threshold=0.01):
    """Removes features with near-zero variance."""
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    mask = selector.get_support()
    print(f"  [FeatureEng] Removed {X.shape[1] - X_filtered.shape[1]} low-variance features")
    return X_filtered, mask


def remove_highly_correlated_features(X, threshold=0.95):
    """Removes one of each pair of features with correlation above threshold."""
    sample_size = min(5000, X.shape[0])
    sample_idx = np.random.choice(X.shape[0], sample_size, replace=False)
    corr_matrix = np.abs(np.corrcoef(X[sample_idx].T))
    upper_tri = np.triu(corr_matrix, k=1)
    to_drop = set()
    for i in range(upper_tri.shape[0]):
        for j in range(i + 1, upper_tri.shape[1]):
            if upper_tri[i, j] > threshold:
                to_drop.add(j)
    retained = [i for i in range(X.shape[1]) if i not in to_drop]
    X_filtered = X[:, retained]
    print(f"  [FeatureEng] Removed {len(to_drop)} highly correlated features (r>{threshold})")
    return X_filtered, retained


def compute_feature_importance(X, y, feature_names=None, top_k=None):
    """Computes feature importance using mutual information + Random Forest."""
    sample_size = min(5000, X.shape[0])
    idx = np.random.choice(X.shape[0], sample_size, replace=False)
    
    mi_scores = mutual_info_classif(X[idx], y[idx], random_state=42, n_neighbors=3)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    importance = dict(zip(feature_names, mi_scores))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    print(f"  [FeatureEng] Top 5 features by Mutual Information:")
    for i, (name, score) in enumerate(list(importance.items())[:5]):
        print(f"    {i+1}. {name}: {score:.4f}")
    
    if top_k and top_k < X.shape[1]:
        top_indices = np.argsort(mi_scores)[-top_k:]
        return X[:, top_indices], importance, top_indices
    
    return X, importance, None


def apply_pca(X, variance_ratio=0.95):
    """Applies PCA for dimensionality reduction."""
    pca = PCA(n_components=variance_ratio, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"  [FeatureEng] PCA: {X.shape[1]} -> {X_pca.shape[1]} components ({variance_ratio*100:.0f}% variance)")
    return X_pca, pca


def run_feature_engineering(X, y, feature_names=None, use_pca=False, top_k_features=None):
    """Runs the complete feature engineering pipeline."""
    print("\n[FeatureEng] === Starting Feature Engineering ===")
    metadata = {}
    
    X, var_mask = remove_low_variance_features(X)
    metadata['variance_mask'] = var_mask
    if feature_names is not None:
        feature_names = [feature_names[i] for i, keep in enumerate(var_mask) if keep]
    
    X, retained = remove_highly_correlated_features(X)
    metadata['retained_indices'] = retained
    if feature_names is not None:
        feature_names = [feature_names[i] for i in retained]
    
    X, mi_imp, mi_idx = compute_feature_importance(X, y, feature_names, top_k=top_k_features)
    metadata['mi_importance'] = mi_imp
    if mi_idx is not None and feature_names is not None:
        feature_names = [feature_names[i] for i in mi_idx]
    
    if use_pca:
        X, pca_obj = apply_pca(X)
        metadata['pca'] = pca_obj
    
    metadata['final_feature_names'] = feature_names
    print(f"[FeatureEng] Final shape: {X.shape}")
    print("[FeatureEng] === Feature Engineering Complete ===\n")
    return X, metadata
