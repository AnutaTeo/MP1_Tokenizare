
# Wrapper around threats.py, takes a prompt and returns a structured danger
# report with a clear verdict and labeled threat breakdown.


import json
from pathlib import Path

# Support both relative imports (from GUI) and direct script execution
try:
    from .threats import scan_prompt, load_threat_knowledge_base
except ImportError:
    from threats import scan_prompt, load_threat_knowledge_base

# Define base directory and prompt file path
BASE_DIR = Path(__file__).parent.parent
PROMPT_FILE = BASE_DIR / "data" / "dangerous_prompt_detection.txt"


# Threat category labels
# Maps internal category keys to human-readable display names.

CATEGORY_LABELS = {
    "system_prompt_extraction": "System Prompt Extraction",
    "instruction_override":     "Instruction Override",
    "identity_hijack":          "Identity / Role Hijack",
    "jailbreak_keyword":        "Jailbreak Keyword",
    "restriction_bypass":       "Restriction Bypass",
    "token_injection":          "Token / Template Injection",
    "encoded_payload":          "Encoded Payload",
    "code_execution":           "Code Execution Attempt",
    "indirect_injection":       "Indirect Injection",
    "context_manipulation":     "Context Manipulation",
    "injection_delimiter":      "Injection Delimiter",
    "context_reset":            "Context Reset",
}

UNICODE_THREAT_LABEL = "Unicode Obfuscation"

# Calibrated danger score thresholds from threat detection dataset
VERDICT_THRESHOLDS = [
    (1.3, "BLOCKED"),
    (0.9, "HIGH_RISK"),
    (0.4, "SUSPICIOUS"),
    (0.0, "CLEAN"),
] 

def _build_threat_list(result: dict) -> list:
    """
    Flattens pattern hits, KB matches, and unicode anomalies into a unified
    list of threat dicts, each with a type, label, severity, and evidence.
    """
    threats = []
    seen_categories = set()

    # Pattern-based threats (deduplicated by category)
    for hit in result["pattern_matches"]:
        cat = hit["category"]
        if cat in seen_categories:
            continue
        seen_categories.add(cat)
        threats.append({
            "type":     "pattern",
            "label":    CATEGORY_LABELS.get(cat, cat.replace("_", " ").title()),
            "severity": hit["severity"],
            "evidence": hit["matched_text"],
            "weight":   hit["weight"],
        })

    # KB statistical phrase matches (top 5 only to keep output clean)
    for match in result["kb_matches"][:5]:
        threats.append({
            "type":     "statistical",
            "label":    "Learned Malicious Phrase",
            "severity": _kb_severity(match["malice_score"]),
            "evidence": match["phrase"],
            "score":    match["malice_score"],
        })

    # Unicode anomalies
    ur = result["unicode_report"]
    if ur["invisible_found"] or ur["homoglyphs_found"] or ur["removed_chars"]:
        evidence_parts = []
        if ur["invisible_found"]:
            evidence_parts.append(f"{len(ur['invisible_found'])} invisible char(s)")
        if ur["homoglyphs_found"]:
            examples = ", ".join(f"'{h['original']}'->'{h['mapped_to']}'" for h in ur["homoglyphs_found"][:3])
            evidence_parts.append(f"homoglyphs: {examples}")
        if ur["removed_chars"]:
            evidence_parts.append(f"{len(ur['removed_chars'])} non-Latin char(s) stripped")
        threats.append({
            "type":     "unicode",
            "label":    UNICODE_THREAT_LABEL,
            "severity": _unicode_severity(result["components"]["unicode_score"]),
            "evidence": "; ".join(evidence_parts),
        })

    return threats


def _kb_severity(score: float) -> str:
    if score >= 8:
        return "critical"
    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def _unicode_severity(score: float) -> str:
    if score >= 60:
        return "critical"
    if score >= 30:
        return "high"
    if score >= 12:
        return "medium"
    return "low"



VERDICT_STYLE = {
    "BLOCKED":    {"symbol": "[BLOCKED]", "description": "BLOCKED - High-confidence attack detected (score >= 1.3)"},
    "HIGH_RISK":  {"symbol": "[HIGH RISK]", "description": "HIGH RISK - Likely adversarial content (score >= 0.9)"},
    "SUSPICIOUS": {"symbol": "[SUSPICIOUS]", "description": "SUSPICIOUS - Suspicious content detected (score >= 0.4)"},
    "CLEAN":      {"symbol": "[CLEAN]", "description": "CLEAN - No significant threats detected (score < 0.4)"},
}



def detect(
    prompt: str,
    kb_path: str = None,
    verbose: bool = False,
) -> dict:
    """
    Scans a prompt and returns a structured danger report.

    Returns:
        {
            "verdict":       "BLOCKED" | "HIGH_RISK" | "SUSPICIOUS" | "CLEAN",
            "danger_score":  float (0–100),
            "auto_blocked":  bool,
            "description":   str,
            "threats":       list of labeled threat dicts,
            "components":    { pattern_score, kb_score, unicode_score },
            "sanitized_text": str,
        }
    """
    kb = {}
    if kb_path and Path(kb_path).exists():
        kb = load_threat_knowledge_base(kb_path)

    raw = scan_prompt(prompt, knowledge_base=kb, verbose=False)

    threats  = _build_threat_list(raw)
    verdict  = raw["verdict"]
    style    = VERDICT_STYLE.get(verdict, VERDICT_STYLE["CLEAN"])
    
    # Get component scores
    components = raw["components"]
    unicode_score = components["unicode_score"]
    danger_score = raw["danger_score"]
    
    # CRITICAL: Unicode obfuscation at high levels automatically escalates verdict
    # High unicode obfuscation indicates deliberate bypass attempt
    if unicode_score >= 80:
        # Automatic block for heavy unicode obfuscation
        verdict = "BLOCKED"
        danger_score = max(danger_score, 95.0)  # Ensure high danger score
        raw["auto_blocked"] = True
    elif unicode_score >= 60:
        # Automatic HIGH_RISK for moderate unicode obfuscation
        if verdict != "BLOCKED":
            verdict = "HIGH_RISK"
            danger_score = max(danger_score, 85.0)

    style    = VERDICT_STYLE.get(verdict, VERDICT_STYLE["CLEAN"])

    report = {
        "verdict":        verdict,
        "symbol":         style["symbol"],
        "description":    style["description"],
        "danger_score":   danger_score,
        "auto_blocked":   raw["auto_blocked"],
        "threats":        threats,
        "threat_count":   len(threats),
        "components":     components,
        "token_count":    raw["token_count"],
        "sanitized_text": raw["sanitized_text"],
    }

    if verbose:
        _print_detect_report(prompt, report)

    return report



def _print_detect_report(original_prompt: str, report: dict):
    """Print an enhanced threat detection report with detailed analysis."""
    sep = "═" * 70
    
    # Header
    print(f"\n{sep}")
    verdict = report['verdict']
    style = VERDICT_STYLE.get(verdict, VERDICT_STYLE["CLEAN"])
    danger_score = report['danger_score']
    
    print(f"  {style['symbol']} {style['description']}")
    print(f"  Danger Score: {danger_score:.2f}/100.0")
    print(f"{sep}\n")
    
    # Input preview
    preview = original_prompt[:100]
    if len(original_prompt) > 100:
        preview += "..."
    print(f"  Input (first 100 chars):\n  \"{preview}\"\n")
    
    # Component breakdown
    components = report['components']
    print(f"  COMPONENT SCORES:")
    print(f"    Pattern Score    : {components['pattern_score']:6.2f}  (regex-based attack detection)")
    print(f"    KB Score         : {components['kb_score']:6.2f}  (learned malicious phrases)")
    print(f"    Unicode Score    : {components['unicode_score']:6.2f}  (obfuscation & encoding)")
    print(f"    ─────────────────")
    print(f"    Total Danger     : {danger_score:6.2f}  (combined threat level)\n")
    
    # Threshold context
    if danger_score >= 1.3:
        threshold_info = f"  [!] BLOCKING THRESHOLD (>= 1.3): Prompt will be BLOCKED"
    elif danger_score >= 0.9:
        threshold_info = f"  [!] HIGH RISK THRESHOLD (>= 0.9): Requires review before proceeding"
    elif danger_score >= 0.4:
        threshold_info = f"  [!] SUSPICIOUS THRESHOLD (>= 0.4): Minor threats detected, proceed with caution"
    else:
        threshold_info = f"  [OK] SAFE ZONE (< 0.4): No significant threats detected"
    print(threshold_info + "\n")
    
    # Threat detection details
    threats = report.get("threats", [])
    auto_blocked = report.get("auto_blocked", False)
    unicode_score = components.get("unicode_score", 0)
    
    # Check if unicode obfuscation caused the escalation
    unicode_triggered_block = False
    if unicode_score >= 80 and verdict == "BLOCKED":
        unicode_triggered_block = True
        print(f"  [CRITICAL] UNICODE OBFUSCATION ESCALATION: Verdict escalated to BLOCKED")
        print(f"    Unicode Score: {unicode_score:.2f} (>= 80 triggers automatic blocking)")
        print(f"    This prompt uses unicode tricks to bypass safety checks\n")
    elif unicode_score >= 60 and verdict == "HIGH_RISK":
        print(f"  [ALERT] UNICODE OBFUSCATION DETECTED: Verdict escalated to HIGH_RISK")
        print(f"    Unicode Score: {unicode_score:.2f} (>= 60 triggers high-risk escalation)")
        print(f"    This prompt contains unicode-based obfuscation attempts\n")
    
    if auto_blocked and not unicode_triggered_block:
        print(f"  [BLOCK] AUTO-BLOCKED: Attack pattern matched strict safety rules\n")
    
    # Check for unicode threats - these should trigger immediate concern
    unicode_threats = [t for t in threats if t["type"] == "unicode"]
    if unicode_threats and unicode_score < 60:
        # Only show this if unicode didn't already trigger escalation
        print(f"  [ALERT] UNICODE OBFUSCATION DETECTED:")
        for t in unicode_threats:
            print(f"    Evidence: {t['evidence']}")
        print(f"    Action: Unicode-based obfuscation attempts are automatically flagged for review\n")
    
    if threats:
        # Threat type summary (excluding unicode)
        non_unicode_threats = [t for t in threats if t["type"] != "unicode"]
        threat_types = {}
        for t in non_unicode_threats:
            t_type = t["type"]
            threat_types[t_type] = threat_types.get(t_type, 0) + 1
        
        if threat_types:
            print(f"  THREAT SUMMARY ({len(non_unicode_threats)} attack-based threats detected):")
            for threat_type in sorted(threat_types.keys()):
                count = threat_types[threat_type]
                if threat_type == "pattern":
                    print(f"    - Dangerous patterns     : {count} (regex rules)")
                elif threat_type == "statistical":
                    print(f"    - Malicious phrases      : {count} (learned from KB)")
            print()
        
        # Detailed threat list
        if non_unicode_threats:
            print(f"  THREAT DETAILS:")
            print(f"  {'-' * 66}")
            
            for i, t in enumerate(non_unicode_threats, 1):
                severity = t.get("severity", "low").upper()
                label = t["label"]
                evidence = t["evidence"]
                
                # Severity indicator without emoji
                if severity == "CRITICAL":
                    severity_mark = "[!!!]"
                elif severity == "HIGH":
                    severity_mark = "[!!]"
                elif severity == "MEDIUM":
                    severity_mark = "[!]"
                else:
                    severity_mark = "[*]"
                
                print(f"  {i}. {severity_mark} [{severity:8}] {label}")
                print(f"     Evidence: {evidence}")
            
            print(f"  {'-' * 66}\n")
    else:
        print(f"  [OK] No specific threats detected in safety analysis\n")
    
    # Summary statistics
    print(f"  ANALYSIS STATISTICS:")
    print(f"    Token count  : {report.get('token_count', 0)}")
    print(f"    Auto-blocked : {'YES - IMMEDIATE ACTION REQUIRED' if auto_blocked else 'NO'}")
    
    # Calibration context
    print(f"\n  CALIBRATION CONTEXT:")
    print(f"    Detection rate on malicious prompts: 41.82%")
    print(f"    False positive rate on normal prompts: 0.0%")
    print(f"    Based on 275 normal + 385 malicious prompts\n")
    
    print(f"{sep}\n")



if __name__ == "__main__":
    if PROMPT_FILE.exists():
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    else:
        print(f"Error: Prompt file not found at {PROMPT_FILE}")
        exit(1)

    # Use default KB path
    kb_path = BASE_DIR / "knowledge" / "threat_patterns.json"
    
    result = detect(prompt_text, kb_path=str(kb_path), verbose=True)