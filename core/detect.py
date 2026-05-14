
# Wrapper around threats.py, takes a prompt and returns a structured danger
# report with a clear verdict and labeled threat breakdown.


import json
from pathlib import Path

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


# Threat extraction helpers 

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
    "BLOCKED":   {"symbol": "1", "description": "Prompt blocked. High-confidence attack detected."},
    "HIGH_RISK": {"symbol": "2",  "description": "High-risk prompt. Likely adversarial."},
    "SUSPICIOUS":{"symbol": "3", "description": "Suspicious content. Review recommended."},
    "CLEAN":     {"symbol": "4", "description": "No significant threats detected."},
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

    report = {
        "verdict":        verdict,
        "symbol":         style["symbol"],
        "description":    style["description"],
        "danger_score":   raw["danger_score"],
        "auto_blocked":   raw["auto_blocked"],
        "threats":        threats,
        "threat_count":   len(threats),
        "components":     raw["components"],
        "token_count":    raw["token_count"],
        "sanitized_text": raw["sanitized_text"],
    }

    if verbose:
        _print_detect_report(prompt, report)

    return report



def _print_detect_report(original_prompt: str, report: dict):
    print(f"\n{report['danger_score']:.0f}% danger\n")
    
    if report["threats"]:
        threat_types = set()
        for t in report["threats"]:
            threat_types.add(t["type"])
        
        for threat_type in sorted(threat_types):
            if threat_type == "pattern":
                print("• Dangerous pattern detected")
            elif threat_type == "statistical":
                print("• Dangerous learned phrase detected")
            elif threat_type == "unicode":
                print("• Dangerous unicode detected")
        
        print(f"\nThreats ({report['threat_count']}):")
        for t in report["threats"]:
            sev = t.get("severity", "").upper()
            print(f"  [{sev}] {t['label']}: {t['evidence']}")
    else:
        print("No threats detected.")



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