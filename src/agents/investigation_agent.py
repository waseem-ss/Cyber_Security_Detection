"""
Investigation Agent — Alert correlation and attack hypothesis generation.

Correlates multiple alerts, identifies attack campaigns, generates
evidence chains, and provides explainability for each detection.
"""
import numpy as np
from collections import defaultdict
from datetime import datetime


# Known attack patterns/signatures for correlation
ATTACK_PATTERNS = {
    'brute_force_campaign': {
        'attacks': ['FTP-Patator', 'SSH-Patator', 'Web Attack \xe2\x80\x93 Brute Force'],
        'description': 'Credential stuffing / brute force campaign across multiple services',
        'mitigation': 'Implement account lockout, MFA, rate limiting'
    },
    'dos_campaign': {
        'attacks': ['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'DDoS'],
        'description': 'Denial of Service campaign targeting service availability',
        'mitigation': 'Enable DDoS protection, rate limiting, CDN, traffic scrubbing'
    },
    'web_attack_campaign': {
        'attacks': ['Web Attack \xe2\x80\x93 Brute Force', 'Web Attack \xe2\x80\x93 XSS',
                    'Web Attack \xe2\x80\x93 Sql Injection'],
        'description': 'Web application attack campaign exploiting multiple vectors',
        'mitigation': 'WAF rules, input validation, parameterized queries, CSP headers'
    },
    'reconnaissance': {
        'attacks': ['PortScan'],
        'description': 'Network reconnaissance / port scanning activity',
        'mitigation': 'Monitor for follow-up attacks, block scanning IPs, IDS rules'
    },
    'advanced_persistent_threat': {
        'attacks': ['Infiltration', 'Bot', 'Heartbleed'],
        'description': 'Possible APT with data exfiltration or backdoor installation',
        'mitigation': 'Immediate isolation, forensic analysis, incident response team'
    }
}


class InvestigationAgent:
    """
    Agent responsible for correlating alerts, generating attack hypotheses,
    and providing evidence-based explanations for detections.
    """
    
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.investigations = []
        self.investigation_id = 0
    
    def investigate(self, alerts):
        """
        Investigate a batch of alerts from the Detection Agent.
        
        Args:
            alerts: list of alert dicts from DetectionAgent
        
        Returns:
            list of investigation report dicts
        """
        if not alerts:
            return []
        
        # Group alerts by attack type
        by_type = defaultdict(list)
        for alert in alerts:
            by_type[alert['predicted_attack']].append(alert)
        
        reports = []
        
        # Identify attack campaigns
        campaigns = self._identify_campaigns(by_type)
        
        # Generate investigation report for each alert group
        for attack_type, type_alerts in by_type.items():
            self.investigation_id += 1
            
            report = {
                'investigation_id': f"INV-{self.investigation_id:06d}",
                'timestamp': datetime.now().isoformat(),
                'attack_type': attack_type,
                'alert_count': len(type_alerts),
                'severity': max(type_alerts, key=lambda x: 
                    {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(x['severity'], 0)
                )['severity'],
                'avg_confidence': np.mean([a['confidence'] for a in type_alerts]),
                'model_agreement_rate': sum(1 for a in type_alerts if a['model_agreement']) / len(type_alerts),
                'related_campaigns': [c['campaign_name'] for c in campaigns 
                                     if attack_type in c['attack_types']],
                'hypothesis': self._generate_hypothesis(attack_type, type_alerts),
                'evidence': self._collect_evidence(type_alerts),
                'recommended_action': self._get_recommendation(attack_type),
                'needs_escalation': any(a['severity'] in ['CRITICAL', 'HIGH'] for a in type_alerts),
                'low_confidence_alerts': [a['alert_id'] for a in type_alerts if a['needs_investigation']]
            }
            reports.append(report)
        
        self.investigations.extend(reports)
        return reports
    
    def _identify_campaigns(self, alerts_by_type):
        """Identify multi-vector attack campaigns from correlated alerts."""
        campaigns = []
        detected_types = set(alerts_by_type.keys())
        
        for pattern_name, pattern_info in ATTACK_PATTERNS.items():
            matching_types = detected_types.intersection(set(pattern_info['attacks']))
            if len(matching_types) >= 1:
                campaigns.append({
                    'campaign_name': pattern_name,
                    'attack_types': list(matching_types),
                    'description': pattern_info['description'],
                    'mitigation': pattern_info['mitigation'],
                    'multi_vector': len(matching_types) > 1
                })
        
        return campaigns
    
    def _generate_hypothesis(self, attack_type, alerts):
        """Generate an attack hypothesis based on the detected pattern."""
        confidence = np.mean([a['confidence'] for a in alerts])
        count = len(alerts)
        
        hypotheses = {
            'DDoS': f"Distributed Denial of Service attack detected with {count} malicious flows "
                    f"(avg confidence: {confidence:.1%}). Likely volumetric attack attempting "
                    f"to overwhelm network resources.",
            'DoS Hulk': f"HTTP flooding attack (Hulk) detected. {count} flows exhibit "
                       f"characteristics of randomized HTTP GET/POST floods.",
            'DoS GoldenEye': f"GoldenEye HTTP DoS attack detected. {count} flows show "
                            f"keep-alive connection abuse patterns.",
            'DoS slowloris': f"Slowloris attack detected. {count} flows show partial HTTP request "
                            f"patterns designed to exhaust connection pools.",
            'DoS Slowhttptest': f"Slow HTTP POST attack detected. {count} flows exhibit "
                               f"slow body transmission patterns.",
            'PortScan': f"Port scanning activity detected across {count} flows. "
                       f"Reconnaissance phase — expect follow-up exploitation attempts.",
            'Bot': f"Botnet activity detected. {count} flows show automated command-and-control "
                  f"communication patterns.",
            'Infiltration': f"Network infiltration detected. {count} flows show data exfiltration "
                           f"or lateral movement patterns. IMMEDIATE ACTION REQUIRED.",
            'Heartbleed': f"Heartbleed (CVE-2014-0160) exploitation attempt. {count} flows show "
                         f"TLS heartbeat extension abuse. CRITICAL VULNERABILITY.",
            'FTP-Patator': f"FTP brute force attack detected. {count} flows show rapid "
                          f"authentication attempts against FTP service.",
            'SSH-Patator': f"SSH brute force attack detected. {count} flows show rapid "
                          f"authentication attempts against SSH service.",
        }
        
        default_hyp = (f"Attack type '{attack_type}' detected with {count} suspicious flows "
                      f"(avg confidence: {confidence:.1%}).")
        
        return hypotheses.get(attack_type, default_hyp)
    
    def _collect_evidence(self, alerts):
        """Collect evidence supporting the detection."""
        return {
            'total_alerts': len(alerts),
            'high_confidence_alerts': sum(1 for a in alerts if a['confidence'] > 0.9),
            'models_agreed': sum(1 for a in alerts if a['model_agreement']),
            'max_confidence': max(a['confidence'] for a in alerts),
            'min_confidence': min(a['confidence'] for a in alerts),
            'anomaly_scores': [a['anomaly_score'] for a in alerts if a.get('anomaly_score')]
        }
    
    def _get_recommendation(self, attack_type):
        """Get recommended response action for the attack type."""
        recommendations = {
            'DDoS': 'BLOCK source IPs, enable DDoS mitigation, alert NOC',
            'DoS Hulk': 'Rate limit HTTP connections, block source, enable WAF rules',
            'DoS GoldenEye': 'Limit keep-alive connections, block source, restart affected services',
            'DoS slowloris': 'Set connection timeouts, limit concurrent connections per IP',
            'DoS Slowhttptest': 'Set request body timeouts, enable mod_reqtimeout',
            'PortScan': 'Monitor for follow-up attacks, update firewall rules',
            'Bot': 'Isolate infected hosts, block C2 domains, malware scan',
            'Infiltration': 'ISOLATE affected systems, forensic capture, incident response',
            'Heartbleed': 'Patch OpenSSL immediately, revoke certificates, rotate keys',
            'FTP-Patator': 'Enable account lockout, block source IP, review FTP access',
            'SSH-Patator': 'Enable fail2ban, block source IP, enforce key-based auth',
        }
        return recommendations.get(attack_type, 'Investigate further and update detection rules')
    
    def get_summary(self):
        """Returns investigation summary."""
        if not self.investigations:
            return "No investigations conducted."
        return {
            'total_investigations': len(self.investigations),
            'escalation_needed': sum(1 for r in self.investigations if r['needs_escalation']),
            'attack_types_found': list(set(r['attack_type'] for r in self.investigations)),
        }
