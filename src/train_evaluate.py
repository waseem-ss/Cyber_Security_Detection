"""
Comprehensive Training and Evaluation Module.
Handles class-weighted training, focal loss, confusion matrix heatmaps,
ROC curves, and detailed per-class analysis for all attack types.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    auc, f1_score, roc_auc_score, roc_curve
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf


def train_model(model, X_train, y_train, X_val, y_val,
                class_weights=None, batch_size=128, epochs=30,
                model_name='best_model', use_focal_loss=False):
    """
    Trains the model with class weights, LR scheduling, and early stopping.
    """
    os.makedirs('saved_models', exist_ok=True)
    save_path = f'saved_models/{model_name}.keras'
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]
    
    print(f"\nTraining {model.name} for up to {epochs} epochs...")
    if class_weights:
        print(f"  Using class weights for {len(class_weights)} classes")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    print(f"  Model saved to {save_path}")
    return history


def train_autoencoder(autoencoder, X_train_benign, X_val_benign,
                      batch_size=128, epochs=30):
    """
    Trains autoencoder on BENIGN traffic only for anomaly detection.
    """
    os.makedirs('saved_models', exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint('saved_models/autoencoder.keras', monitor='val_loss',
                       save_best_only=True, verbose=1),
    ]
    
    print(f"\nTraining Autoencoder on BENIGN traffic ({X_train_benign.shape[0]} samples)...")
    history = autoencoder.fit(
        X_train_benign, X_train_benign,
        validation_data=(X_val_benign, X_val_benign),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history


def evaluate_model(model, X_test, y_test, label_encoder, model_name='model'):
    """
    Comprehensive evaluation with per-class metrics, confusion matrix, and ROC.
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")
    
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    target_names = [str(cls) for cls in label_encoder.classes_]
    
    os.makedirs('artifacts', exist_ok=True)
    report_path = f'artifacts/evaluation_{model_name}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"  Evaluation Report: {model_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Test Loss: {loss:.4f}\nTest Accuracy: {accuracy:.4f}\n\n")
        
        report = classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        
        # PR-AUC per class
        num_classes = len(label_encoder.classes_)
        y_test_oh = np.eye(num_classes)[y_test]
        
        f.write("PR-AUC per class:\n")
        for i, cls_name in enumerate(target_names):
            if np.sum(y_test_oh[:, i]) > 0:
                prec, rec, _ = precision_recall_curve(y_test_oh[:, i], y_pred_probs[:, i])
                pr_auc_val = auc(rec, prec)
                f.write(f"  {cls_name:>40s}: {pr_auc_val:.4f}\n")
        
        # Attack detection summary
        f.write("\n\nAttack Detection Summary:\n")
        f.write("-" * 60 + "\n")
        for i, cls_name in enumerate(target_names):
            cls_mask = (y_test == i)
            if np.sum(cls_mask) > 0:
                cls_pred = y_pred[cls_mask]
                detected = np.sum(cls_pred == i)
                total = np.sum(cls_mask)
                recall_val = detected / total * 100
                status = "OK" if recall_val > 50 else "CRITICAL"
                f.write(f"  {cls_name:>40s}: {detected}/{total} detected "
                        f"({recall_val:.1f}%) [{status}]\n")
    
    print(f"  Report saved to {report_path}")
    
    # Generate visualizations
    _plot_confusion_matrix(y_test, y_pred, target_names, model_name)
    _plot_roc_curves(y_test, y_pred_probs, target_names, model_name)
    _plot_training_safe(model_name)
    
    return y_pred, y_pred_probs


def evaluate_autoencoder(autoencoder, X_test_flat, y_test, label_encoder,
                         threshold_percentile=95):
    """
    Evaluates autoencoder anomaly detection capability.
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: Autoencoder (Anomaly Detection)")
    print(f"{'='*60}")
    
    reconstructions = autoencoder.predict(X_test_flat, verbose=0)
    mse = np.mean(np.square(X_test_flat - reconstructions), axis=1)
    
    # Binary labels: 0=BENIGN, 1=ATTACK
    benign_idx = list(label_encoder.classes_).index('BENIGN') if 'BENIGN' in label_encoder.classes_ else 0
    y_binary = (y_test != benign_idx).astype(int)
    
    # Determine threshold from benign reconstruction errors
    benign_mse = mse[y_test == benign_idx]
    threshold = np.percentile(benign_mse, threshold_percentile)
    
    y_pred_ae = (mse > threshold).astype(int)
    
    # Per-class anomaly scores
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/evaluation_autoencoder.txt', 'w') as f:
        f.write("Autoencoder Anomaly Detection Results\n")
        f.write(f"Threshold (p{threshold_percentile}): {threshold:.6f}\n\n")
        
        from sklearn.metrics import precision_score, recall_score, f1_score as f1
        prec = precision_score(y_binary, y_pred_ae, zero_division=0)
        rec = recall_score(y_binary, y_pred_ae, zero_division=0)
        f1_val = f1(y_binary, y_pred_ae, zero_division=0)
        
        f.write(f"Binary (BENIGN vs ATTACK):\n")
        f.write(f"  Precision: {prec:.4f}\n  Recall: {rec:.4f}\n  F1: {f1_val:.4f}\n\n")
        
        f.write("Mean Reconstruction Error per Class:\n")
        for i, cls_name in enumerate(label_encoder.classes_):
            cls_mask = (y_test == i)
            if np.sum(cls_mask) > 0:
                cls_mse = np.mean(mse[cls_mask])
                detected = np.sum(y_pred_ae[cls_mask])
                total = np.sum(cls_mask)
                f.write(f"  {str(cls_name):>40s}: MSE={cls_mse:.6f} | "
                        f"Detected={detected}/{total}\n")
    
    print(f"  Autoencoder report saved to artifacts/evaluation_autoencoder.txt")
    
    # Plot reconstruction error distribution
    plt.figure(figsize=(10, 5))
    plt.hist(mse[y_binary == 0], bins=100, alpha=0.6, label='BENIGN', color='green', density=True)
    plt.hist(mse[y_binary == 1], bins=100, alpha=0.6, label='ATTACK', color='red', density=True)
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold={threshold:.4f}')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.title('Autoencoder: Reconstruction Error Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('artifacts/autoencoder_distribution.png', dpi=150)
    plt.close()
    
    return y_pred_ae, mse


def _plot_confusion_matrix(y_test, y_pred, target_names, model_name):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    fig_size = max(8, len(target_names) * 0.8)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                linewidths=0.5)
    plt.title(f'Confusion Matrix (Normalized) — {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f'artifacts/confusion_matrix_{model_name}.png', dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to artifacts/confusion_matrix_{model_name}.png")


def _plot_roc_curves(y_test, y_pred_probs, target_names, model_name):
    """Generates per-class ROC curves."""
    num_classes = len(target_names)
    y_test_oh = np.eye(num_classes)[y_test]
    
    plt.figure(figsize=(10, 8))
    for i, cls_name in enumerate(target_names):
        if np.sum(y_test_oh[:, i]) > 0 and np.sum(y_test_oh[:, i]) < len(y_test):
            fpr, tpr, _ = roc_curve(y_test_oh[:, i], y_pred_probs[:, i])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{cls_name} (AUC={roc_auc_val:.3f})', alpha=0.7)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves — {model_name}')
    plt.legend(fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.savefig(f'artifacts/roc_curves_{model_name}.png', dpi=150)
    plt.close()
    print(f"  ROC curves saved to artifacts/roc_curves_{model_name}.png")


def plot_training_history(history, model_name='model'):
    """Plots training/validation loss and accuracy."""
    os.makedirs('artifacts', exist_ok=True)
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} — Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} — Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'artifacts/training_history_{model_name}.png', dpi=150)
    plt.close()
    print(f"  Training history saved to artifacts/training_history_{model_name}.png")


def _plot_training_safe(model_name):
    """Placeholder — actual plot called separately via plot_training_history."""
    pass
