"""
Response Agent — Automated countermeasures based on threat severity.

Implements policy-driven responses with playbooks for each attack category,
simulated defensive actions, and comprehensive audit logging.
"""
from datetime import datetime
import json
import os


# Response playbooks per attack type
RESPONSE_PLAYBOOKS = {
    'DDoS': {
        'actions': ['block_source_ips', 'enable_ddos_mitigation', 'alert_noc', 'increase_capacity'],
        'auto_execute': True,
        'human_approval_needed': False,
        'description': 'Automated DDoS mitigation sequence'
    },
    'DoS Hulk': {
        'actions': ['rate_limit_http', 'block_source_ip', 'enable_waf_rules'],
        'auto_execute': True,
        'human_approval_needed': False,
        'description': 'HTTP flood mitigation'
    },
    'DoS GoldenEye': {
        'actions': ['limit_keepalive', 'block_source_ip', 'restart_service'],
        'auto_execute': True,
        'human_approval_needed': False,
        'description': 'GoldenEye attack mitigation'
    },
    'DoS slowloris': {
        'actions': ['set_connection_timeout', 'limit_concurrent_connections', 'block_source_ip'],
        'auto_execute': True,
        'human_approval_needed': False,
        'description': 'Slowloris attack mitigation'
    },
    'DoS Slowhttptest': {
        'actions': ['set_body_timeout', 'enable_mod_reqtimeout', 'block_source_ip'],
        'auto_execute': True,
        'human_approval_needed': False,
        'description': 'Slow HTTP attack mitigation'
    },
    'PortScan': {
        'actions': ['log_scanning_activity', 'update_firewall_rules', 'increase_monitoring'],
        'auto_execute': True,
        'human_approval_needed': False,
        'description': 'Reconnaissance response'
    },
    'Bot': {
        'actions': ['isolate_host', 'block_c2_domains', 'malware_scan', 'alert_soc'],
        'auto_execute': False,
        'human_approval_needed': True,
        'description': 'Botnet containment — requires SOC approval'
    },
    'Infiltration': {
        'actions': ['isolate_systems', 'forensic_capture', 'block_lateral_movement',
                    'incident_response_team', 'preserve_evidence'],
        'auto_execute': False,
        'human_approval_needed': True,
        'description': 'APT/Infiltration response — REQUIRES IMMEDIATE HUMAN APPROVAL'
    },
    'Heartbleed': {
        'actions': ['patch_openssl', 'revoke_certificates', 'rotate_keys',
                    'block_exploit_traffic', 'alert_security_team'],
        'auto_execute': False,
        'human_approval_needed': True,
        'description': 'Critical vulnerability exploitation — REQUIRES HUMAN APPROVAL'
    },
    'FTP-Patator': {
        'actions': ['enable_account_lockout', 'block_source_ip', 'review_ftp_access'],
        'auto_execute': True,
        'human_approval_needed': False,
        'description': 'FTP brute force mitigation'
    },
    'SSH-Patator': {
        'actions': ['enable_fail2ban', 'block_source_ip', 'enforce_key_auth'],
        'auto_execute': True,
        'human_approval_needed': False,
        'description': 'SSH brute force mitigation'
    },
    'Web Attack \xe2\x80\x93 Brute Force': {
        'actions': ['enable_captcha', 'rate_limit_login', 'block_source_ip', 'alert_app_team'],
        'auto_execute': True,
        'human_approval_needed': False,
        'description': 'Web login brute force mitigation'
    },
    'Web Attack \xe2\x80\x93 XSS': {
        'actions': ['enable_csp_headers', 'waf_xss_rules', 'sanitize_inputs', 'alert_dev_team'],
        'auto_execute': True,
        'human_approval_needed': False,
        'description': 'XSS attack mitigation'
    },
    'Web Attack \xe2\x80\x93 Sql Injection': {
        'actions': ['waf_sqli_rules', 'parameterize_queries', 'block_source_ip',
                    'database_audit', 'alert_dev_team'],
        'auto_execute': False,
        'human_approval_needed': True,
        'description': 'SQL Injection mitigation — check for data exfiltration'
    }
}


class ResponseAgent:
    """
    Agent responsible for executing automated countermeasures based on
    investigation reports and predefined response playbooks.
    """
    
    def __init__(self):
        self.response_log = []
        self.response_id = 0
    
    def respond(self, investigation_reports):
        """
        Execute response playbooks based on investigation reports.
        
        Args:
            investigation_reports: list of investigation report dicts
        
        Returns:
            list of response action dicts
        """
        responses = []
        
        for report in investigation_reports:
            attack_type = report['attack_type']
            severity = report['severity']
            
            playbook = RESPONSE_PLAYBOOKS.get(attack_type, {
                'actions': ['log_and_monitor', 'alert_soc'],
                'auto_execute': False,
                'human_approval_needed': True,
                'description': f'Generic response for {attack_type}'
            })
            
            self.response_id += 1
            
            response = {
                'response_id': f"RSP-{self.response_id:06d}",
                'timestamp': datetime.now().isoformat(),
                'investigation_id': report['investigation_id'],
                'attack_type': attack_type,
                'severity': severity,
                'playbook': playbook['description'],
                'actions_taken': [],
                'human_approval_needed': playbook['human_approval_needed'],
                'status': 'PENDING' if playbook['human_approval_needed'] else 'EXECUTED'
            }
            
            # Simulate action execution
            for action in playbook['actions']:
                action_result = self._execute_action(action, attack_type, playbook['auto_execute'])
                response['actions_taken'].append(action_result)
            
            responses.append(response)
        
        self.response_log.extend(responses)
        return responses
    
    def _execute_action(self, action, attack_type, auto_execute):
        """Simulate execution of a defensive action."""
        return {
            'action': action,
            'status': 'EXECUTED' if auto_execute else 'AWAITING_APPROVAL',
            'timestamp': datetime.now().isoformat(),
            'details': f"[SIM] {action} triggered for {attack_type}"
        }
    
    def generate_audit_log(self, output_dir='artifacts'):
        """Generate comprehensive audit log of all response actions."""
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, 'response_audit_log.txt')
        
        with open(log_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("  RESPONSE AGENT — AUDIT LOG\n")
            f.write("=" * 70 + "\n\n")
            
            for resp in self.response_log:
                f.write(f"Response ID: {resp['response_id']}\n")
                f.write(f"  Timestamp: {resp['timestamp']}\n")
                f.write(f"  Investigation: {resp['investigation_id']}\n")
                f.write(f"  Attack Type: {resp['attack_type']}\n")
                f.write(f"  Severity: {resp['severity']}\n")
                f.write(f"  Playbook: {resp['playbook']}\n")
                f.write(f"  Status: {resp['status']}\n")
                f.write(f"  Human Approval Needed: {resp['human_approval_needed']}\n")
                f.write(f"  Actions:\n")
                for act in resp['actions_taken']:
                    f.write(f"    - {act['action']}: {act['status']}\n")
                f.write("\n" + "-" * 40 + "\n\n")
        
        print(f"  Audit log saved to {log_path}")
        return log_path
    
    def get_summary(self):
        """Returns response summary."""
        if not self.response_log:
            return "No responses executed."
        return {
            'total_responses': len(self.response_log),
            'auto_executed': sum(1 for r in self.response_log if r['status'] == 'EXECUTED'),
            'pending_approval': sum(1 for r in self.response_log if r['status'] == 'PENDING'),
            'total_actions': sum(len(r['actions_taken']) for r in self.response_log),
        }
