# Parses a dataset of prompts (one per line) and computes the average number of
# threat signals per prompt — regex pattern hits, KB phrase matches, and unicode

import argparse
import json
import math
import statistics
from pathlib import Path

from threats import (
    scan_prompt,
    load_threat_knowledge_base,
    VERDICT_THRESHOLDS,
)

def load_lines(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def extract_signal_counts(result: dict) -> dict:
    ur = result["unicode_report"]
    return {
        "pattern_hits":     len(result["pattern_matches"]),
        "kb_hits":          len(result["kb_matches"]),
        "invisible_chars":  len(ur["invisible_found"]),
        "homoglyphs":       len(ur["homoglyphs_found"]),
        "removed_chars":    len(ur["removed_chars"]),
        "total_unicode":    len(ur["invisible_found"]) + len(ur["homoglyphs_found"]) + len(ur["removed_chars"]),
        "danger_score":     result["danger_score"],
        "pattern_score":    result["components"]["pattern_score"],
        "kb_score":         result["components"]["kb_score"],
        "unicode_score":    result["components"]["unicode_score"],
        "auto_blocked":     int(result["auto_blocked"]),
    }


def _safe_stats(values: list) -> dict:
    if not values:
        return {"mean": 0, "median": 0, "stdev": 0, "min": 0, "max": 0, "p90": 0}
    sorted_v = sorted(values)
    p90_idx  = max(0, math.ceil(len(sorted_v) * 0.9) - 1)
    return {
        "mean":   round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "stdev":  round(statistics.stdev(values) if len(values) > 1 else 0.0, 3),
        "min":    round(min(values), 3),
        "max":    round(max(values), 3),
        "p90":    round(sorted_v[p90_idx], 3),
    }

def compute_corpus_stats(signals: list) -> dict:
    keys = list(signals[0].keys()) if signals else []
    return {k: _safe_stats([s[k] for s in signals]) for k in keys}


def recommend_thresholds(normal_stats: dict, malicious_stats: dict) -> dict:
    """
    Suggests danger_score thresholds by finding the score band that
    cleanly separates normal p90 from malicious p10.

    Strategy:
      - CLEAN upper bound   = normal  danger_score p90   (stay below this for clean prompts)
      - BLOCKED lower bound = malicious danger_score p10 (everything above this is likely malicious)
      - SUSPICIOUS / HIGH_RISK split the gap linearly
    """
    normal_p90    = normal_stats["danger_score"]["p90"]
    mal_p10_idx   = 0  # p10 computed inline below — we use mean - stdev as a conservative floor
    mal_mean      = malicious_stats["danger_score"]["mean"]
    mal_stdev     = malicious_stats["danger_score"]["stdev"]
    mal_floor     = max(round(mal_mean - mal_stdev, 1), normal_p90 + 1)

    gap           = 100.0 - mal_floor
    suspicious    = round(normal_p90 + (mal_floor - normal_p90) * 0.33, 1)
    high_risk     = round(normal_p90 + (mal_floor - normal_p90) * 0.66, 1)
    blocked       = round(mal_floor, 1)

    return {
        "CLEAN_upper_bound": round(normal_p90, 1),
        "recommended": [
            {"label": "BLOCKED",   "min_score": blocked},
            {"label": "HIGH_RISK", "min_score": high_risk},
            {"label": "SUSPICIOUS","min_score": suspicious},
            {"label": "CLEAN",     "min_score": 0},
        ],
        "notes": (
            f"Normal p90={normal_p90}, malicious floor={mal_floor}. "
            f"Gap split into thirds: SUSPICIOUS>={suspicious}, HIGH_RISK>={high_risk}, BLOCKED>={blocked}."
        ),
    }


def evaluate_current_thresholds(normal_signals: list, malicious_signals: list) -> dict:
    def _classify(score):
        for threshold, label in VERDICT_THRESHOLDS:
            if score >= threshold:
                return label
        return "CLEAN"

    normal_verdicts    = [_classify(s["danger_score"]) for s in normal_signals]
    malicious_verdicts = [_classify(s["danger_score"]) for s in malicious_signals]

    def _count(verdicts):
        counts = {"CLEAN": 0, "SUSPICIOUS": 0, "HIGH_RISK": 0, "BLOCKED": 0}
        for v in verdicts:
            counts[v] = counts.get(v, 0) + 1
        total = len(verdicts) or 1
        return {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in counts.items()}

    return {
        "normal_verdict_distribution":    _count(normal_verdicts),
        "malicious_verdict_distribution": _count(malicious_verdicts),
        "false_positive_rate":  round(sum(1 for v in normal_verdicts    if v != "CLEAN") / max(len(normal_verdicts), 1) * 100, 2),
        "detection_rate":       round(sum(1 for v in malicious_verdicts if v != "CLEAN") / max(len(malicious_verdicts), 1) * 100, 2),
        "block_rate":           round(sum(1 for v in malicious_verdicts if v == "BLOCKED") / max(len(malicious_verdicts), 1) * 100, 2),
    }



def calibrate(
    malicious_path: str,
    normal_path: str,
    kb_path: str = None,
    output_path: str = "knowledge/calibration_report.json",
) -> dict:
    kb = load_threat_knowledge_base(kb_path) if kb_path and Path(kb_path).exists() else {}

    print(f"Loading normal prompts from    : {normal_path}")
    print(f"Loading malicious prompts from : {malicious_path}")
    if kb:
        print(f"Knowledge base loaded          : {len(kb)} patterns")
    else:
        print("Knowledge base                 : not loaded (KB scoring disabled)")

    normal_lines    = load_lines(normal_path)
    malicious_lines = load_lines(malicious_path)

    print(f"\nScanning {len(normal_lines)} normal prompts...")
    normal_signals = [extract_signal_counts(scan_prompt(p, kb)) for p in normal_lines]

    print(f"Scanning {len(malicious_lines)} malicious prompts...")
    malicious_signals = [extract_signal_counts(scan_prompt(p, kb)) for p in malicious_lines]

    normal_stats    = compute_corpus_stats(normal_signals)
    malicious_stats = compute_corpus_stats(malicious_signals)

    thresholds      = recommend_thresholds(normal_stats, malicious_stats)
    evaluation      = evaluate_current_thresholds(normal_signals, malicious_signals)

    report = {
        "dataset_sizes": {
            "normal":    len(normal_lines),
            "malicious": len(malicious_lines),
        },
        "normal_stats":           normal_stats,
        "malicious_stats":        malicious_stats,
        "recommended_thresholds": thresholds,
        "current_threshold_evaluation": evaluation,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 56)
    print("  CALIBRATION REPORT")
    print("=" * 56)
    print(f"\n  Normal   danger_score : mean={normal_stats['danger_score']['mean']:6.1f}  p90={normal_stats['danger_score']['p90']:6.1f}")
    print(f"  Malicious danger_score: mean={malicious_stats['danger_score']['mean']:6.1f}  p90={malicious_stats['danger_score']['p90']:6.1f}")
    print(f"\n  Average signals per NORMAL prompt:")
    print(f"    pattern_hits   : {normal_stats['pattern_hits']['mean']}")
    print(f"    kb_hits        : {normal_stats['kb_hits']['mean']}")
    print(f"    unicode_events : {normal_stats['total_unicode']['mean']}")
    print(f"\n  Average signals per MALICIOUS prompt:")
    print(f"    pattern_hits   : {malicious_stats['pattern_hits']['mean']}")
    print(f"    kb_hits        : {malicious_stats['kb_hits']['mean']}")
    print(f"    unicode_events : {malicious_stats['total_unicode']['mean']}")
    print(f"\n  Current threshold evaluation:")
    print(f"    False positive rate : {evaluation['false_positive_rate']}%")
    print(f"    Detection rate      : {evaluation['detection_rate']}%")
    print(f"    Block rate          : {evaluation['block_rate']}%")
    print(f"\n  Recommended thresholds:")
    for t in thresholds["recommended"]:
        print(f"    {t['label']:12} >= {t['min_score']}")
    print(f"\n  {thresholds['notes']}")
    print(f"\n  Full report saved to: {output_path}")
    print("=" * 56)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate threat detection thresholds.")
    parser.add_argument("--malicious", default="data/malicious_prompts.txt")
    parser.add_argument("--normal",    default="data/normal_prompts.txt")
    parser.add_argument("--kb",        default="knowledge/threat_patterns.json")
    parser.add_argument("--output",    default="knowledge/calibration_report.json")
    args = parser.parse_args()

    calibrate(
        malicious_path=args.malicious,
        normal_path=args.normal,
        kb_path=args.kb,
        output_path=args.output,
    )