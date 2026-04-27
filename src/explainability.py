"""
Explainability Module (XAI) for Cyber Security Threat Detection.

Provides feature importance analysis and decision audit trails
for transparent and interpretable threat detection decisions.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def compute_permutation_importance(model, X_test, y_test, feature_names=None,
                                    n_repeats=5, top_k=20):
    """
    Compute permutation importance for the model.
    Measures how much accuracy drops when each feature is randomly shuffled.
    
    Args:
        model: trained Keras model
        X_test: test features (samples, timesteps, features)
        y_test: test labels
        feature_names: list of feature names
        n_repeats: number of permutation repeats
        top_k: number of top features to display
    
    Returns:
        dict: feature importance scores
    """
    print("\n[XAI] Computing permutation importance...")
    
    # Baseline accuracy
    y_pred_base = np.argmax(model.predict(X_test, verbose=0), axis=1)
    base_accuracy = np.mean(y_pred_base == y_test)
    
    n_features = X_test.shape[2]  # (samples, timesteps, features)
    importances = np.zeros(n_features)
    
    for feat_idx in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, :, feat_idx])
            y_pred_perm = np.argmax(model.predict(X_permuted, verbose=0), axis=1)
            perm_accuracy = np.mean(y_pred_perm == y_test)
            drops.append(base_accuracy - perm_accuracy)
        importances[feat_idx] = np.mean(drops)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    importance_dict = dict(zip(feature_names, importances))
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    # Plot top features
    top_features = list(importance_dict.items())[:top_k]
    
    os.makedirs('artifacts', exist_ok=True)
    plt.figure(figsize=(10, 6))
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    plt.barh(range(len(names)), values, color='steelblue')
    plt.yticks(range(len(names)), names, fontsize=7)
    plt.xlabel('Accuracy Drop (Permutation Importance)')
    plt.title(f'Top {top_k} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('artifacts/feature_importance.png', dpi=150)
    plt.close()
    print(f"  Feature importance plot saved to artifacts/feature_importance.png")
    
    return importance_dict


def generate_decision_audit(alerts, investigations, output_dir='artifacts'):
    """
    Generate a human-readable decision audit trail explaining
    why each threat was detected and what actions were taken.
    """
    os.makedirs(output_dir, exist_ok=True)
    audit_path = os.path.join(output_dir, 'decision_audit_trail.txt')
    
    with open(audit_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  DECISION AUDIT TRAIL — Explainable AI Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Alerts Analyzed: {len(alerts)}\n")
        f.write(f"Total Investigations: {len(investigations)}\n\n")
        
        for inv in investigations:
            f.write(f"\n{'─' * 60}\n")
            f.write(f"Investigation: {inv['investigation_id']}\n")
            f.write(f"Attack Type: {inv['attack_type']}\n")
            f.write(f"Severity: {inv['severity']}\n")
            f.write(f"Confidence: {inv['avg_confidence']:.1%}\n")
            f.write(f"Model Agreement: {inv['model_agreement_rate']:.1%}\n")
            f.write(f"\nHypothesis:\n  {inv['hypothesis']}\n")
            f.write(f"\nEvidence:\n")
            evidence = inv['evidence']
            f.write(f"  - Total alerts: {evidence['total_alerts']}\n")
            f.write(f"  - High confidence: {evidence['high_confidence_alerts']}\n")
            f.write(f"  - Models agreed: {evidence['models_agreed']}\n")
            f.write(f"  - Confidence range: {evidence['min_confidence']:.3f} — "
                    f"{evidence['max_confidence']:.3f}\n")
            f.write(f"\nRecommended Action:\n  {inv['recommended_action']}\n")
            if inv['related_campaigns']:
                f.write(f"\nRelated Campaigns: {', '.join(inv['related_campaigns'])}\n")
            if inv['needs_escalation']:
                f.write(f"\n⚠ ESCALATION REQUIRED — Human approval needed\n")
    
    print(f"  Decision audit trail saved to {audit_path}")
    return audit_path
