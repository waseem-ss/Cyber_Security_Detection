"""
Coordinator Agent — Orchestrates Detection, Investigation, and Response agents.

Manages inter-agent communication, priority assignment, escalation workflows,
and human-in-the-loop decisions for high-risk actions.
"""
from datetime import datetime
import os


class CoordinatorAgent:
    """
    Central orchestrator that coordinates the Detection, Investigation,
    and Response agents. Manages the complete threat lifecycle.
    """
    
    def __init__(self, detection_agent, investigation_agent, response_agent):
        self.detection = detection_agent
        self.investigation = investigation_agent
        self.response = response_agent
        self.pipeline_log = []
        self.escalations = []
    
    def run_pipeline(self, X_batch, batch_ids=None):
        """
        Execute the full agent pipeline: Detect → Investigate → Respond.
        
        Args:
            X_batch: network flow features
            batch_ids: optional flow identifiers
        
        Returns:
            dict with complete pipeline results
        """
        pipeline_start = datetime.now()
        
        # Phase 1: Detection
        print("\n  [Coordinator] Phase 1: Detection Agent analyzing flows...")
        alerts = self.detection.analyze(X_batch, batch_ids)
        print(f"  [Coordinator] Detection complete: {len(alerts)} alerts generated")
        
        if not alerts:
            result = {
                'timestamp': pipeline_start.isoformat(),
                'status': 'CLEAN',
                'message': 'No threats detected in this batch',
                'flows_analyzed': X_batch.shape[0],
                'alerts': 0,
                'investigations': 0,
                'responses': 0
            }
            self.pipeline_log.append(result)
            return result
        
        # Phase 2: Investigation
        print("  [Coordinator] Phase 2: Investigation Agent correlating alerts...")
        investigations = self.investigation.investigate(alerts)
        print(f"  [Coordinator] Investigation complete: {len(investigations)} reports")
        
        # Phase 3: Response
        print("  [Coordinator] Phase 3: Response Agent executing playbooks...")
        responses = self.response.respond(investigations)
        print(f"  [Coordinator] Response complete: {len(responses)} actions")
        
        # Check for escalations
        escalation_needed = [inv for inv in investigations if inv['needs_escalation']]
        if escalation_needed:
            print(f"  [Coordinator] [WARNING] {len(escalation_needed)} investigations need escalation!")
            self.escalations.extend(escalation_needed)
        
        result = {
            'timestamp': pipeline_start.isoformat(),
            'status': 'THREATS_DETECTED',
            'flows_analyzed': X_batch.shape[0],
            'alerts': len(alerts),
            'investigations': len(investigations),
            'responses': len(responses),
            'escalations': len(escalation_needed),
            'detection_summary': self.detection.get_summary(),
            'investigation_summary': self.investigation.get_summary(),
            'response_summary': self.response.get_summary()
        }
        
        self.pipeline_log.append(result)
        return result
    
    def generate_comprehensive_report(self, output_dir='artifacts'):
        """Generate a comprehensive report of all agent activities."""
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'agent_pipeline_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("  AGENTIC AI CYBER SECURITY — COMPREHENSIVE REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Detection Summary
            det_summary = self.detection.get_summary()
            f.write("DETECTION AGENT SUMMARY\n")
            f.write("-" * 40 + "\n")
            if isinstance(det_summary, dict):
                f.write(f"  Total Alerts: {det_summary['total_alerts']}\n")
                f.write(f"  Low Confidence (needs review): {det_summary['low_confidence_count']}\n")
                f.write(f"  By Severity:\n")
                for sev, count in sorted(det_summary.get('by_severity', {}).items()):
                    f.write(f"    {sev}: {count}\n")
                f.write(f"  By Attack Type:\n")
                for atk, count in sorted(det_summary.get('by_attack_type', {}).items()):
                    f.write(f"    {atk}: {count}\n")
            else:
                f.write(f"  {det_summary}\n")
            
            # Investigation Summary
            f.write(f"\nINVESTIGATION AGENT SUMMARY\n")
            f.write("-" * 40 + "\n")
            inv_summary = self.investigation.get_summary()
            if isinstance(inv_summary, dict):
                f.write(f"  Total Investigations: {inv_summary['total_investigations']}\n")
                f.write(f"  Escalation Needed: {inv_summary['escalation_needed']}\n")
                f.write(f"  Attack Types: {', '.join(inv_summary['attack_types_found'])}\n")
            else:
                f.write(f"  {inv_summary}\n")
            
            # Investigation Details
            f.write(f"\nINVESTIGATION DETAILS\n")
            f.write("-" * 40 + "\n")
            for inv in self.investigation.investigations:
                f.write(f"\n  [{inv['investigation_id']}] {inv['attack_type']}\n")
                f.write(f"    Severity: {inv['severity']}\n")
                f.write(f"    Alerts: {inv['alert_count']}\n")
                f.write(f"    Confidence: {inv['avg_confidence']:.1%}\n")
                f.write(f"    Model Agreement: {inv['model_agreement_rate']:.1%}\n")
                f.write(f"    Hypothesis: {inv['hypothesis'][:200]}...\n")
                f.write(f"    Action: {inv['recommended_action']}\n")
                if inv['related_campaigns']:
                    f.write(f"    Campaigns: {', '.join(inv['related_campaigns'])}\n")
            
            # Response Summary
            f.write(f"\nRESPONSE AGENT SUMMARY\n")
            f.write("-" * 40 + "\n")
            resp_summary = self.response.get_summary()
            if isinstance(resp_summary, dict):
                f.write(f"  Total Responses: {resp_summary['total_responses']}\n")
                f.write(f"  Auto-Executed: {resp_summary['auto_executed']}\n")
                f.write(f"  Pending Approval: {resp_summary['pending_approval']}\n")
                f.write(f"  Total Actions: {resp_summary['total_actions']}\n")
            else:
                f.write(f"  {resp_summary}\n")
            
            # Escalations
            if self.escalations:
                f.write(f"\n{'!'*70}\n")
                f.write(f"  ESCALATIONS REQUIRING HUMAN ATTENTION\n")
                f.write(f"{'!'*70}\n")
                for esc in self.escalations:
                    f.write(f"\n  [{esc['investigation_id']}] {esc['attack_type']} "
                           f"- Severity: {esc['severity']}\n")
                    f.write(f"    {esc['hypothesis'][:200]}\n")
        
        print(f"\n  Comprehensive report saved to {report_path}")
        
        # Also generate audit log
        self.response.generate_audit_log(output_dir)
        
        return report_path
