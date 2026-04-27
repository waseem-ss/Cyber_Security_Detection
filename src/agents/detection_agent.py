"""
Detection Agent — Core threat detection using ensemble models.

Uses CNN-LSTM, Transformer, and Autoencoder models to classify incoming
network flows, generate alerts with confidence scores, and flag low-confidence
predictions for the Investigation Agent.
"""
import numpy as np
from datetime import datetime


class DetectionAgent:
    """
    Agent responsible for analyzing network traffic and detecting threats.
    Runs multiple models in parallel and aggregates results.
    """
    
    def __init__(self, models, label_encoder, autoencoder=None, confidence_threshold=0.7):
        """
        Args:
            models: dict of {name: keras_model} for classification
            label_encoder: fitted LabelEncoder
            autoencoder: optional autoencoder for anomaly detection
            confidence_threshold: minimum confidence for high-confidence alert
        """
        self.models = models
        self.label_encoder = label_encoder
        self.autoencoder = autoencoder
        self.confidence_threshold = confidence_threshold
        self.alerts = []
        self.alert_id_counter = 0
    
    def analyze(self, X_batch, batch_ids=None):
        """
        Analyze a batch of network flows using all available models.
        
        Args:
            X_batch: input features (samples, timesteps, features)
            batch_ids: optional identifiers for each flow
        
        Returns:
            list of alert dicts
        """
        if batch_ids is None:
            batch_ids = [f"flow_{i}" for i in range(X_batch.shape[0])]
        
        # Get predictions from each model
        all_preds = {}
        all_probs = {}
        for name, model in self.models.items():
            probs = model.predict(X_batch, verbose=0)
            preds = np.argmax(probs, axis=1)
            all_preds[name] = preds
            all_probs[name] = probs
        
        # Ensemble: average probabilities
        avg_probs = np.mean(list(all_probs.values()), axis=0)
        ensemble_preds = np.argmax(avg_probs, axis=1)
        ensemble_confidence = np.max(avg_probs, axis=1)
        
        # Autoencoder anomaly scores
        anomaly_scores = None
        if self.autoencoder is not None:
            X_flat = X_batch.reshape(X_batch.shape[0], -1)
            recon = self.autoencoder.predict(X_flat, verbose=0)
            anomaly_scores = np.mean(np.square(X_flat - recon), axis=1)
        
        # Generate alerts
        batch_alerts = []
        benign_label = 'BENIGN'
        benign_idx = list(self.label_encoder.classes_).index(benign_label) \
            if benign_label in self.label_encoder.classes_ else None
        
        for i in range(X_batch.shape[0]):
            pred_label = self.label_encoder.classes_[ensemble_preds[i]]
            confidence = ensemble_confidence[i]
            
            is_threat = (ensemble_preds[i] != benign_idx) if benign_idx is not None else True
            
            if is_threat:
                self.alert_id_counter += 1
                severity = self._compute_severity(pred_label, confidence, 
                                                   anomaly_scores[i] if anomaly_scores is not None else 0)
                
                alert = {
                    'alert_id': f"ALERT-{self.alert_id_counter:06d}",
                    'timestamp': datetime.now().isoformat(),
                    'flow_id': batch_ids[i],
                    'predicted_attack': pred_label,
                    'confidence': float(confidence),
                    'severity': severity,
                    'anomaly_score': float(anomaly_scores[i]) if anomaly_scores is not None else None,
                    'needs_investigation': confidence < self.confidence_threshold,
                    'model_agreement': self._check_agreement(all_preds, i),
                    'individual_predictions': {
                        name: str(self.label_encoder.classes_[preds[i]])
                        for name, preds in all_preds.items()
                    }
                }
                batch_alerts.append(alert)
        
        self.alerts.extend(batch_alerts)
        return batch_alerts
    
    def _compute_severity(self, attack_type, confidence, anomaly_score):
        """Compute alert severity based on attack type and confidence."""
        critical_attacks = ['Heartbleed', 'Web Attack \xe2\x80\x93 Sql Injection',
                           'Web Attack \xe2\x80\x93 XSS', 'Infiltration']
        high_attacks = ['DDoS', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris',
                       'DoS Slowhttptest', 'Bot']
        medium_attacks = ['PortScan', 'FTP-Patator', 'SSH-Patator',
                         'Web Attack \xe2\x80\x93 Brute Force']
        
        if attack_type in critical_attacks:
            return 'CRITICAL'
        elif attack_type in high_attacks:
            return 'HIGH'
        elif attack_type in medium_attacks:
            return 'MEDIUM'
        else:
            return 'HIGH' if confidence > 0.9 else 'MEDIUM'
    
    def _check_agreement(self, all_preds, idx):
        """Check if all models agree on the prediction."""
        predictions = [preds[idx] for preds in all_preds.values()]
        return len(set(predictions)) == 1
    
    def get_summary(self):
        """Returns a summary of all generated alerts."""
        if not self.alerts:
            return "No threats detected."
        
        summary = {
            'total_alerts': len(self.alerts),
            'by_severity': {},
            'by_attack_type': {},
            'low_confidence_count': sum(1 for a in self.alerts if a['needs_investigation']),
        }
        for alert in self.alerts:
            sev = alert['severity']
            summary['by_severity'][sev] = summary['by_severity'].get(sev, 0) + 1
            atk = alert['predicted_attack']
            summary['by_attack_type'][atk] = summary['by_attack_type'].get(atk, 0) + 1
        
        return summary
