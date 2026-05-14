import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")


# ══════════════════════════════════════════════════════════════
#  SHARED TOKENIZER UTILITIES
#  Same functions as learn_redundant_patterns.py — kept here so
#  threats.py is importable standalone without circular deps.
# ══════════════════════════════════════════════════════════════

def tokenize_text(text: str) -> list:
    return encoding.encode(text)

def token_count(text: str) -> int:
    return len(tokenize_text(text))

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_words(text: str) -> list:
    return re.findall(r"\b\w+\b", text.lower())

def extract_ngrams(words: list, n: int) -> list:
    return [
        " ".join(words[i : i + n])
        for i in range(len(words) - n + 1)
    ]

def build_ngram_frequency_map(lines: list, n: int) -> Counter:
    counter = Counter()
    for line in lines:
        words = extract_words(normalize_text(line))
        counter.update(extract_ngrams(words, n))
    return counter


# ══════════════════════════════════════════════════════════════
#  UNICODE LAYER — allowed ranges, blocked ranges, homoglyphs
# ══════════════════════════════════════════════════════════════

ALLOWED_UNICODE_RANGES = [
    (0x0009, 0x000D),  # Tab, LF, VT, FF, CR
    (0x0020, 0x007E),  # Basic Latin (printable ASCII)
    (0x00A0, 0x00FF),  # Latin-1 Supplement  (à á â ã ä å ç è é ê ë î ï ô ù û ü ...)
    (0x0100, 0x017F),  # Latin Extended-A    (ă Ă ș Ș ț Ț ...)
    (0x0180, 0x024F),  # Latin Extended-B
    (0x0250, 0x02AF),  # IPA Extensions
    (0x02B0, 0x02FF),  # Spacing Modifier Letters
    (0x0300, 0x036F),  # Combining Diacritical Marks
    (0x1E00, 0x1EFF),  # Latin Extended Additional
    (0x2000, 0x206F),  # General Punctuation  (zero-widths removed in pass-1)
    (0x20A0, 0x20CF),  # Currency Symbols
    (0x2100, 0x214F),  # Letterlike Symbols  (TM (R) degC ...)
    (0x2190, 0x22FF),  # Arrows + Math Operators
]

BLOCKED_SCRIPT_RANGES = {
    "greek":          (0x0370, 0x03FF),
    "cyrillic":       (0x0400, 0x04FF),
    "cyrillic_supp":  (0x0500, 0x052F),
    "armenian":       (0x0530, 0x058F),
    "hebrew":         (0x0590, 0x05FF),
    "arabic":         (0x0600, 0x06FF),
    "devanagari":     (0x0900, 0x097F),
    "cjk":            (0x4E00, 0x9FFF),
    "hiragana":       (0x3040, 0x309F),
    "katakana":       (0x30A0, 0x30FF),
    "math_latin":     (0x1D400, 0x1D7FF),  # bold/italic/script letters
    "enclosed_alpha": (0x2460, 0x24FF),    # circled letters
    "fullwidth":      (0xFF00, 0xFFEF),    # fullwidth ASCII
    "runic":          (0x16A0, 0x16FF),
}

INVISIBLE_CHARS = {
    "\u00AD",  # Soft Hyphen
    "\u200B",  # Zero Width Space
    "\u200C",  # Zero Width Non-Joiner
    "\u200D",  # Zero Width Joiner
    "\u200E",  # Left-to-Right Mark
    "\u200F",  # Right-to-Left Mark
    "\u202A",  # Left-to-Right Embedding
    "\u202B",  # Right-to-Left Embedding
    "\u202C",  # Pop Directional Formatting
    "\u202D",  # Left-to-Right Override
    "\u202E",  # Right-to-Left Override
    "\u2060",  # Word Joiner
    "\u2061",  # Function Application
    "\u2062",  # Invisible Times
    "\u2063",  # Invisible Separator
    "\u2064",  # Invisible Plus
    "\u2066",  # Left-to-Right Isolate
    "\u2067",  # Right-to-Left Isolate
    "\u2068",  # First Strong Isolate
    "\u2069",  # Pop Directional Isolate
    "\uFEFF",  # BOM / Zero Width No-Break Space
}

HOMOGLYPH_MAP = {
    # Cyrillic -> Latin
    "\u0430": "a", "\u0435": "e", "\u043E": "o", "\u0440": "r",
    "\u0441": "c", "\u0445": "x", "\u0443": "y", "\u0456": "i",
    "\u0301": "",
    # Greek -> Latin
    "\u03BF": "o", "\u03B1": "a", "\u03B5": "e", "\u03B9": "i",
    "\u03BA": "k", "\u03BD": "v", "\u03C1": "p", "\u03C5": "u",
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z",
    "\u0397": "H", "\u0399": "I", "\u039A": "K", "\u039C": "M",
    "\u039D": "N", "\u039F": "O", "\u03A1": "P", "\u03A4": "T",
    "\u03A5": "Y", "\u03A7": "X",
    # Fullwidth -> ASCII
    **{chr(0xFF01 + i): chr(0x21 + i) for i in range(94)},
}

# Per-signal danger weights feeding into unicode_score
UNICODE_WEIGHTS = {
    "invisible_char":  8,   # per invisible char found
    "homoglyph":       12,  # per homoglyph substitution
    "blocked_script":  20,  # per non-Latin char removed
}


# ══════════════════════════════════════════════════════════════
#  REGEX THREAT PATTERNS
#  weight feeds directly into pattern_score (0-100).
# ══════════════════════════════════════════════════════════════

_RAW_THREAT_PATTERNS = [
    # System prompt extraction
    {
        "pattern":    r"(reveal|show|print|output|display|repeat|tell me|give me|what (is|are))\s+(your\s+)?(system\s*prompt|instructions|initial\s*prompt|base\s*prompt|hidden\s*instructions)",
        "severity":   "critical", "category": "system_prompt_extraction",
        "auto_block": True,  "weight": 100,
    },
    {
        "pattern":    r"(ignore|disregard|forget|bypass|override|skip)\s+(all\s+)?(previous|prior|above|your|the)?\s*(instructions?|prompt|rules?|constraints?|guidelines?|context|training)",
        "severity":   "critical", "category": "instruction_override",
        "auto_block": True,  "weight": 100,
    },
    {
        "pattern":    r"what\s+(were\s+you|was\s+your)\s+(told|instructed|given|asked|trained)",
        "severity":   "high",     "category": "system_prompt_extraction",
        "auto_block": True,  "weight": 85,
    },
    # Identity / role hijack
    {
        "pattern":    r"(you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be)|roleplay\s+as|simulate\s+(being|a)|imagine\s+you\s+are|behave\s+as|from\s+now\s+on\s+you\s+(are|will))",
        "severity":   "high",     "category": "identity_hijack",
        "auto_block": False, "weight": 65,
    },
    {
        "pattern":    r"\b(DAN|jailbreak|evil\s*mode|developer\s*mode|god\s*mode|unrestricted\s*mode|MalwareGPT|EvilBOT|rogue\s*ai)\b",
        "severity":   "critical", "category": "jailbreak_keyword",
        "auto_block": True,  "weight": 100,
    },
    {
        "pattern":    r"(your\s+true\s+self|without\s+(any\s+)?(restrictions?|filters?|limits?|rules?)|no\s+(restrictions?|filters?|limits?|censorship)|forget\s+(ethical|your)\s+(guidelines?|training|alignment))",
        "severity":   "high",     "category": "restriction_bypass",
        "auto_block": True,  "weight": 90,
    },
    # Token / template injection
    {
        "pattern":    r"<\s*(system|assistant|user|im_start|im_end|instruction|prompt)\s*>",
        "severity":   "critical", "category": "token_injection",
        "auto_block": True,  "weight": 100,
    },
    {
        "pattern":    r"\[INST\]|\[\/INST\]|\[SYS\]|\[\/SYS\]|<<SYS>>|<</SYS>>",
        "severity":   "critical", "category": "token_injection",
        "auto_block": True,  "weight": 100,
    },
    # Encoding tricks
    {
        "pattern":    r"base64\s*[:\-]?\s*[A-Za-z0-9+/]{20,}={0,2}",
        "severity":   "high",     "category": "encoded_payload",
        "auto_block": True,  "weight": 85,
    },
    {
        "pattern":    r"(execute|run|eval)\s+(this|the\s+following|below)?\s*(code|script|command|payload)",
        "severity":   "high",     "category": "code_execution",
        "auto_block": True,  "weight": 85,
    },
    # Indirect injection
    {
        "pattern":    r"(note\s+to\s+(the\s+)?(ai|llm|model|assistant|system)|attention\s*:\s*(ai|llm|model))",
        "severity":   "high",     "category": "indirect_injection",
        "auto_block": True,  "weight": 80,
    },
    {
        "pattern":    r"(everything\s+(above|before|prior)\s+was\s+(fake|a\s+test|not\s+real))",
        "severity":   "high",     "category": "context_manipulation",
        "auto_block": True,  "weight": 85,
    },
    # Delimiter injection
    {
        "pattern":    r"(^|\n)\s*#{3,}",
        "severity":   "medium",   "category": "injection_delimiter",
        "auto_block": False, "weight": 30,
    },
    {
        "pattern":    r"(^|\n)\s*-{4,}",
        "severity":   "medium",   "category": "injection_delimiter",
        "auto_block": False, "weight": 30,
    },
    # Context reset
    {
        "pattern":    r"(start|begin|new)\s+(a\s+)?(fresh|new|clean)\s+(conversation|session|context|chat)|reset\s+(your\s+)?(memory|context|instructions?)",
        "severity":   "medium",   "category": "context_reset",
        "auto_block": False, "weight": 40,
    },
]

THREAT_PATTERNS = [
    {**e, "pattern": re.compile(e["pattern"], re.IGNORECASE | re.MULTILINE)}
    for e in _RAW_THREAT_PATTERNS
]


# ══════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE — TRAINING
#
#  Re-uses the same log-ratio + length-weight formula from
#  learn_redundant_patterns.py, but trained on
#  (normal_prompts, malicious_prompts) instead of
#  (normal_prompts, redundant_prompts).
# ══════════════════════════════════════════════════════════════

def compute_malice_scores(
    normal_freq: Counter,
    malicious_freq: Counter,
    min_malicious_frequency: int = 2,
    threshold: float = 1.5,
) -> dict:
    scores = {}

    for phrase, mal_count in malicious_freq.items():
        if mal_count < min_malicious_frequency:
            continue
        if len(phrase.split()) < 2:
            continue  # unigrams too noisy

        normal_count = normal_freq.get(phrase, 0)
        ngram_len    = len(phrase.split())

        base_score     = math.log(mal_count + 1) / math.log(normal_count + 2)
        weighted_score = base_score * ngram_len   # longer phrases rank higher

        if weighted_score >= threshold:
            scores[phrase] = {
                "type":                "malicious_phrase",
                "source":              "statistical",
                "malice_score":        round(weighted_score, 3),
                "base_score":          round(base_score, 3),
                "length_multiplier":   ngram_len,
                "normal_frequency":    normal_count,
                "malicious_frequency": mal_count,
                "token_cost":          token_count(phrase),
                "ngram_size":          ngram_len,
            }

    return scores


def build_threat_knowledge_base(
    normal_lines: list,
    malicious_lines: list,
    ngram_sizes: list = None,
) -> dict:
    if ngram_sizes is None:
        ngram_sizes = [2, 3, 4, 5]

    knowledge = {}

    for n in ngram_sizes:
        print(f"  Scanning {n}-grams for malicious patterns...")

        normal_freq    = build_ngram_frequency_map(normal_lines,    n)
        malicious_freq = build_ngram_frequency_map(malicious_lines, n)
        scores         = compute_malice_scores(normal_freq, malicious_freq)

        # Keep highest-scoring entry across passes (same as learn_redundant_patterns.py)
        for phrase, data in scores.items():
            if (
                phrase not in knowledge
                or data["malice_score"] > knowledge[phrase]["malice_score"]
            ):
                knowledge[phrase] = data

    return knowledge


def save_threat_knowledge_base(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_threat_knowledge_base(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
#  UNICODE SANITIZATION
# ══════════════════════════════════════════════════════════════

def _in_allowed_range(cp: int) -> bool:
    for start, end in ALLOWED_UNICODE_RANGES:
        if start <= cp <= end:
            return True
    return False


def sanitize_unicode(text: str) -> dict:
    removed_chars    = []
    homoglyphs_found = []
    invisible_found  = []

    # Pass 1 — strip invisible / direction-override chars
    step1 = []
    for ch in text:
        if ch in INVISIBLE_CHARS:
            invisible_found.append(ch)
        else:
            step1.append(ch)

    # Pass 2 — normalise homoglyphs to ASCII equivalents
    step2 = []
    for ch in step1:
        if ch in HOMOGLYPH_MAP:
            mapped = HOMOGLYPH_MAP[ch]
            homoglyphs_found.append({"original": ch, "mapped_to": mapped})
            step2.append(mapped)
        else:
            step2.append(ch)

    # Pass 3 — whitelist: drop anything outside allowed Latin ranges
    step3 = []
    for ch in step2:
        cp = ord(ch)
        if _in_allowed_range(cp):
            step3.append(ch)
        else:
            name = unicodedata.name(ch, f"U+{cp:04X}")
            cat  = unicodedata.category(ch)
            removed_chars.append({"char": ch, "name": name, "category": cat})

    return {
        "sanitized":        "".join(step3),
        "invisible_found":  [repr(c) for c in invisible_found],
        "homoglyphs_found": homoglyphs_found,
        "removed_chars":    removed_chars,
    }


# ══════════════════════════════════════════════════════════════
#  SCORING COMPONENTS
# ══════════════════════════════════════════════════════════════

def compute_unicode_score(ur: dict) -> float:
    """0-100. Weighted count of unicode attack signals."""
    raw = (
        len(ur["invisible_found"])  * UNICODE_WEIGHTS["invisible_char"]
        + len(ur["homoglyphs_found"]) * UNICODE_WEIGHTS["homoglyph"]
        + len(ur["removed_chars"])    * UNICODE_WEIGHTS["blocked_script"]
    )
    return round(min(raw, 100.0), 2)


def compute_pattern_score(threats: list) -> float:
    """
    0-100. Highest matched weight dominates; additional hits add a
    capped soft bonus so multiple medium hits accumulate.
    """
    if not threats:
        return 0.0
    weights = [t["weight"] for t in threats]
    max_w   = max(weights)
    bonus   = min(sum(w for w in weights if w < max_w) * 0.10, 20.0)
    return round(min(max_w + bonus, 100.0), 2)


def compute_kb_score(text: str, knowledge_base: dict) -> tuple:
    """
    0-100. Extracts all ngrams, looks each up in the threat KB.
    Longest/highest-scoring phrase dominates; sub-phrases add
    a diminishing bonus to avoid double-counting.

    Returns (score: float, matched_phrases: list).
    """
    if not knowledge_base:
        return 0.0, []

    words        = extract_words(normalize_text(text))
    matched      = []
    seen_phrases = set()

    for n in range(2, 6):
        for gram in extract_ngrams(words, n):
            if gram in knowledge_base and gram not in seen_phrases:
                entry = knowledge_base[gram]
                matched.append({
                    "phrase":       gram,
                    "malice_score": entry["malice_score"],
                    "token_cost":   entry["token_cost"],
                    "ngram_size":   entry["ngram_size"],
                })
                seen_phrases.add(gram)

    if not matched:
        return 0.0, []

    matched.sort(key=lambda x: (-x["ngram_size"], -x["malice_score"]))
    top_score = matched[0]["malice_score"]

    # Normalise: scores above 10 are very strong signals → cap maps to 100
    base  = min((top_score / 10.0) * 100, 100.0)
    extra = sum(min(m["malice_score"] / 10.0 * 15, 10.0) for m in matched[1:])

    return round(min(base + extra, 100.0), 2), matched


# ══════════════════════════════════════════════════════════════
#  COMBINED DANGER SCORE
# ══════════════════════════════════════════════════════════════

SCORE_WEIGHTS = {
    "pattern": 0.45,  # regex layer — most precise, highest trust
    "kb":      0.35,  # statistical KB — catches learned phrasing
    "unicode": 0.20,  # obfuscation attempts
}

VERDICT_THRESHOLDS = [
    (80, "BLOCKED"),
    (55, "HIGH_RISK"),
    (25, "SUSPICIOUS"),
    (0,  "CLEAN"),
]


def compute_danger_score(
    unicode_score: float,
    pattern_score: float,
    kb_score: float,
    auto_block: bool,
) -> float:
    blended = (
        pattern_score * SCORE_WEIGHTS["pattern"]
        + kb_score    * SCORE_WEIGHTS["kb"]
        + unicode_score * SCORE_WEIGHTS["unicode"]
    )
    if auto_block:
        blended = max(blended, 95.0)
    return round(min(blended, 100.0), 2)


def _verdict(score: float) -> str:
    for threshold, label in VERDICT_THRESHOLDS:
        if score >= threshold:
            return label
    return "CLEAN"


# ══════════════════════════════════════════════════════════════
#  MAIN SCAN ENTRY POINT
# ══════════════════════════════════════════════════════════════

def scan_prompt(
    text: str,
    knowledge_base: dict = None,
    verbose: bool = False,
) -> dict:
    """
    Full pipeline:
      1. Unicode sanitization
      2. Regex pattern matching on sanitised text
      3. KB ngram scoring on sanitised text
      4. Blend into danger_score

    Returns a structured dict ready for JSON serialisation.
    """
    if knowledge_base is None:
        knowledge_base = {}

    # Step 1: unicode
    unicode_report = sanitize_unicode(text)
    clean_text     = unicode_report["sanitized"]
    unicode_score  = compute_unicode_score(unicode_report)

    # Step 2: regex patterns
    threat_hits  = []
    auto_blocked = False

    for entry in THREAT_PATTERNS:
        match = entry["pattern"].search(clean_text)
        if match:
            threat_hits.append({
                "category":    entry["category"],
                "severity":    entry["severity"],
                "auto_block":  entry["auto_block"],
                "weight":      entry["weight"],
                "matched_text": match.group(0).strip(),
            })
            if entry["auto_block"]:
                auto_blocked = True

    pattern_score = compute_pattern_score(threat_hits)

    # Step 3: KB ngram scoring
    kb_score, kb_matches = compute_kb_score(clean_text, knowledge_base)

    # Step 4: blend
    danger_score = compute_danger_score(
        unicode_score, pattern_score, kb_score, auto_blocked
    )
    verdict = _verdict(danger_score)

    result = {
        "danger_score":    danger_score,
        "verdict":         verdict,
        "auto_blocked":    auto_blocked,
        "components": {
            "pattern_score": pattern_score,
            "kb_score":      kb_score,
            "unicode_score": unicode_score,
        },
        "score_weights":   SCORE_WEIGHTS,
        "token_count":     token_count(text),
        "sanitized_text":  clean_text,
        "unicode_report":  unicode_report,
        "pattern_matches": threat_hits,
        "kb_matches":      kb_matches,
    }

    if verbose:
        _print_report(result)

    return result


# ══════════════════════════════════════════════════════════════
#  VERBOSE REPORT PRINTER
# ══════════════════════════════════════════════════════════════

def _print_report(r: dict):
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  DANGER SCORE : {r['danger_score']:6.1f} / 100   [{r['verdict']}]")
    print(sep)
    c = r["components"]
    print(f"  pattern_score : {c['pattern_score']:6.1f}  (weight {SCORE_WEIGHTS['pattern']})")
    print(f"  kb_score      : {c['kb_score']:6.1f}  (weight {SCORE_WEIGHTS['kb']})")
    print(f"  unicode_score : {c['unicode_score']:6.1f}  (weight {SCORE_WEIGHTS['unicode']})")

    ur = r["unicode_report"]
    if ur["invisible_found"]:
        print(f"\n  Invisible chars stripped : {ur['invisible_found']}")
    if ur["homoglyphs_found"]:
        print(f"\n  Homoglyphs normalised :")
        for h in ur["homoglyphs_found"]:
            print(f"    '{h['original']}' -> '{h['mapped_to']}'")
    if ur["removed_chars"]:
        print(f"\n  Non-Latin chars removed :")
        for rc in ur["removed_chars"]:
            print(f"    '{rc['char']}' ({rc['name']})")

    if r["pattern_matches"]:
        print(f"\n  Regex pattern hits :")
        for t in r["pattern_matches"]:
            flag = "[BLOCK]" if t["auto_block"] else "[WARN] "
            print(f"    {flag} [{t['severity'].upper():8}] {t['category']}")
            print(f"             \"{t['matched_text']}\"")

    if r["kb_matches"]:
        print(f"\n  KB phrase matches (top 10) :")
        for m in r["kb_matches"][:10]:
            print(f"    score={m['malice_score']:6.2f}  tokens={m['token_cost']}  \"{m['phrase']}\"")

    print(f"\n  Sanitized : {r['sanitized_text']}")
    print(sep)


# ══════════════════════════════════════════════════════════════
#  TRAINING ENTRY POINT
#  python threats.py --train
# ══════════════════════════════════════════════════════════════

def train(
    normal_path:    str = "data/normal_prompts.txt",
    malicious_path: str = "data/malicious_prompts.txt",
    output_path:    str = "knowledge/threat_patterns.json",
):
    def _load(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    print("Loading datasets...")
    normal_lines    = _load(normal_path)
    malicious_lines = _load(malicious_path)

    print("Building threat knowledge base...")
    kb = build_threat_knowledge_base(normal_lines, malicious_lines)

    print(f"Saving -> {output_path}  ({len(kb)} patterns)")
    save_threat_knowledge_base(kb, output_path)
    return kb


# ══════════════════════════════════════════════════════════════
#  TRAINING ENTRYPOINT
#  python threats.py
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
