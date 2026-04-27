"""
Agentic-AI Cybersecurity Threat Detection — Main Pipeline

Complete orchestration pipeline that:
1. Loads ALL CICIDS2017 attack types with class-balance handling
2. Performs feature engineering
3. Trains CNN-LSTM, Transformer, and Autoencoder models
4. Builds ensemble predictions
5. Runs multi-agent framework (Detection → Investigation → Response)
6. Generates comprehensive evaluation and XAI reports
"""
import os
import sys
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import load_and_preprocess_data
from src.feature_engineering import run_feature_engineering
from src.model import (
    build_cnn_lstm_model, build_transformer_model,
    build_autoencoder_model, build_ensemble_model, FocalLoss
)
from src.train_evaluate import (
    train_model, train_autoencoder, evaluate_model,
    evaluate_autoencoder, plot_training_history
)
from src.agents.detection_agent import DetectionAgent
from src.agents.investigation_agent import InvestigationAgent
from src.agents.response_agent import ResponseAgent
from src.agents.coordinator_agent import CoordinatorAgent
from src.explainability import compute_permutation_importance, generate_decision_audit


def main():
    print("=" * 70)
    print("  AGENTIC-AI CYBERSECURITY THREAT DETECTION PIPELINE")
    print("=" * 70)
    
    pipeline_start = time.time()
    
    # ========================
    # CONFIGURATION
    # ========================
    dataset_dir = 'kaggle_dataset'
    sample_per_file = 3000      # BENIGN samples per file (all attacks kept)
    epochs = 5                  # Training epochs (early stopping will cut short if needed)
    batch_size = 512
    
    # ========================
    # STEP 1: DATA LOADING
    # ========================
    print("\n" + "=" * 60)
    print("  STEP 1: Loading & Preprocessing ALL CICIDS2017 Data")
    print("=" * 60)
    
    t0 = time.time()
    
    result = load_and_preprocess_data(
        dataset_dir=dataset_dir,
        sample_per_file=sample_per_file,
        min_class_samples=5
    )
    
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     label_encoder, num_features,
     feature_names, class_weights) = result
    
    num_classes = len(label_encoder.classes_)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    print(f"\n  Classes ({num_classes}): {list(label_encoder.classes_)}")
    print(f"  Input shape: {input_shape}")
    print(f"  [TIME] Step 1 completed in {time.time()-t0:.1f}s")
    
    # ========================
    # STEP 2: FEATURE ENGINEERING (on flat data for analysis)
    # ========================
    print("\n" + "=" * 60)
    print("  STEP 2: Feature Engineering & Selection")
    print("=" * 60)
    
    t0 = time.time()
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    _, feat_metadata = run_feature_engineering(
        X_train_flat, y_train, feature_names=feature_names,
        use_pca=False, top_k_features=None
    )
    # Note: We use the full feature set for model training (feature selection is informational)
    print(f"  [TIME] Step 2 completed in {time.time()-t0:.1f}s")
    
    # ========================
    # STEP 3: MODEL TRAINING
    # ========================
    print("\n" + "=" * 60)
    print("  STEP 3: Training Models")
    print("=" * 60)
    
    t0 = time.time()
    
    # --- 3a. CNN-LSTM ---
    print("\n--- 3a: CNN-LSTM Model ---")
    cnn_lstm = build_cnn_lstm_model(input_shape, num_classes)
    cnn_lstm.summary()
    
    history_cnn = train_model(
        cnn_lstm, X_train, y_train, X_val, y_val,
        class_weights=class_weights, batch_size=batch_size,
        epochs=epochs, model_name='cnn_lstm'
    )
    plot_training_history(history_cnn, 'CNN_LSTM')
    
    # --- 3b. Transformer ---
    print("\n--- 3b: Transformer Model ---")
    transformer = build_transformer_model(input_shape, num_classes)
    transformer.summary()
    
    history_tf = train_model(
        transformer, X_train, y_train, X_val, y_val,
        class_weights=class_weights, batch_size=batch_size,
        epochs=epochs, model_name='transformer'
    )
    plot_training_history(history_tf, 'Transformer')
    
    # --- 3c. Autoencoder (trained on BENIGN only) ---
    print("\n--- 3c: Autoencoder (Anomaly Detection) ---")
    benign_label = 'BENIGN'
    benign_idx = list(label_encoder.classes_).index(benign_label) \
        if benign_label in label_encoder.classes_ else 0
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    X_train_benign = X_train_flat[y_train == benign_idx]
    X_val_benign = X_val_flat[y_val == benign_idx]
    
    autoencoder, encoder = build_autoencoder_model(
        input_dim=X_train_flat.shape[1], encoding_dim=32
    )
    autoencoder.summary()
    
    history_ae = train_autoencoder(
        autoencoder, X_train_benign, X_val_benign,
        batch_size=batch_size, epochs=epochs
    )
    print(f"  [TIME] Step 3 completed in {time.time()-t0:.1f}s")
    
    # ========================
    # STEP 4: EVALUATION
    # ========================
    print("\n" + "=" * 60)
    print("  STEP 4: Comprehensive Evaluation")
    print("=" * 60)
    
    t0 = time.time()
    
    # Evaluate CNN-LSTM
    y_pred_cnn, y_probs_cnn = evaluate_model(
        cnn_lstm, X_test, y_test, label_encoder, 'CNN_LSTM'
    )
    
    # Evaluate Transformer
    y_pred_tf, y_probs_tf = evaluate_model(
        transformer, X_test, y_test, label_encoder, 'Transformer'
    )
    
    # Evaluate Autoencoder
    evaluate_autoencoder(autoencoder, X_test_flat, y_test, label_encoder)
    
    # Ensemble evaluation (average probabilities)
    print(f"\n{'='*60}")
    print(f"  Evaluating: Ensemble (CNN-LSTM + Transformer)")
    print(f"{'='*60}")
    
    ensemble_probs = (y_probs_cnn + y_probs_tf) / 2.0
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    from sklearn.metrics import classification_report
    target_names = [str(cls) for cls in label_encoder.classes_]
    report = classification_report(y_test, ensemble_preds,
                                   target_names=target_names, zero_division=0)
    
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/evaluation_ensemble.txt', 'w') as f:
        f.write("Ensemble Model (CNN-LSTM + Transformer) — Classification Report\n\n")
        f.write(report)
    print(f"  Ensemble report saved to artifacts/evaluation_ensemble.txt")
    print(report)
    print(f"  [TIME] Step 4 completed in {time.time()-t0:.1f}s")
    
    # ========================
    # STEP 5: AGENTIC AI FRAMEWORK
    # ========================
    print("\n" + "=" * 60)
    print("  STEP 5: Multi-Agent Cyber Defense Simulation")
    print("=" * 60)
    
    t0 = time.time()
    
    # Initialize agents
    detection_agent = DetectionAgent(
        models={'CNN_LSTM': cnn_lstm, 'Transformer': transformer},
        label_encoder=label_encoder,
        autoencoder=autoencoder,
        confidence_threshold=0.7
    )
    investigation_agent = InvestigationAgent(feature_names=feature_names)
    response_agent = ResponseAgent()
    coordinator = CoordinatorAgent(detection_agent, investigation_agent, response_agent)
    
    # Run agent pipeline on a sample of test data (attack samples only)
    attack_mask = (y_test != benign_idx)
    attack_indices = np.where(attack_mask)[0]
    
    # Take a sample of attack flows for demonstration
    sample_size = min(100, len(attack_indices))
    sample_idx = np.random.choice(attack_indices, sample_size, replace=False)
    X_agent_test = X_test[sample_idx]
    
    pipeline_result = coordinator.run_pipeline(X_agent_test)
    
    print(f"\n  Pipeline Result:")
    print(f"    Status: {pipeline_result['status']}")
    print(f"    Flows Analyzed: {pipeline_result['flows_analyzed']}")
    print(f"    Alerts: {pipeline_result.get('alerts', 0)}")
    print(f"    Investigations: {pipeline_result.get('investigations', 0)}")
    print(f"    Responses: {pipeline_result.get('responses', 0)}")
    print(f"    Escalations: {pipeline_result.get('escalations', 0)}")
    
    # Generate comprehensive report
    coordinator.generate_comprehensive_report()
    print(f"  [TIME] Step 5 completed in {time.time()-t0:.1f}s")
    
    # ========================
    # STEP 6: EXPLAINABILITY (XAI)
    # ========================
    print("\n" + "=" * 60)
    print("  STEP 6: Explainability (XAI)")
    print("=" * 60)
    
    t0 = time.time()
    
    # Feature importance (on a small test subset for speed)
    xai_sample = min(500, X_test.shape[0])
    xai_idx = np.random.choice(X_test.shape[0], xai_sample, replace=False)
    compute_permutation_importance(
        cnn_lstm, X_test[xai_idx], y_test[xai_idx],
        feature_names=feature_names, n_repeats=2, top_k=15
    )
    
    # Decision audit trail
    generate_decision_audit(
        detection_agent.alerts,
        investigation_agent.investigations
    )
    print(f"  [TIME] Step 6 completed in {time.time()-t0:.1f}s")
    
    # ========================
    # FINAL SUMMARY
    # ========================
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    total_time = time.time() - pipeline_start
    print(f"\n  Total pipeline time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\n  Models trained: CNN-LSTM, Transformer, Autoencoder")
    print(f"  Attack types covered: {num_classes - 1} (+ BENIGN)")
    print(f"  Classes: {list(label_encoder.classes_)}")
    print(f"\n  Artifacts generated:")
    for f in sorted(os.listdir('artifacts')):
        fpath = os.path.join('artifacts', f)
        size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
        print(f"    - {f} ({size:,} bytes)")
    print(f"\n  Saved models:")
    for f in sorted(os.listdir('saved_models')):
        fpath = os.path.join('saved_models', f)
        size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
        print(f"    - {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
