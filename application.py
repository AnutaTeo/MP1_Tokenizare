import json
import re
import tkinter as tk
import unicodedata
from pathlib import Path
from tkinter import messagebox, filedialog
from tkinter import ttk
import tiktoken

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge" / "redundant_patterns.json"
DEFAULT_PROMPT_FILE = BASE_DIR / "data" / "prompt_to_optimize.txt"

encoding = tiktoken.get_encoding("cl100k_base")

try:
    from core import threats as external_threats
except Exception:
    external_threats = None


def load_knowledge_base(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        messagebox.showerror(
            "Error",
            f"Knowledge base file not found:\n{path}"
        )
        return {}

def normalize_spaces(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    return text.strip()

def normalize_phrase(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def count_tokens(text):
    return len(encoding.encode(text))

def get_tokens(text):
    token_ids = encoding.encode(text)
    decoded_tokens = []
    for token_id in token_ids:
        token_bytes = encoding.decode_single_token_bytes(token_id)

        try:
            token_text = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_text = str(token_bytes)

        decoded_tokens.append(token_text)

    return token_ids, decoded_tokens

def get_score(data):
    score1 = data.get("redundancy_score", 0)
    score2 = data.get("score", 0)

    try:
        score1 = float(score1)
    except (TypeError, ValueError):
        score1 = 0

    try:
        score2 = float(score2)
    except (TypeError, ValueError):
        score2 = 0

    return max(score1, score2)

def replace_first_occurrence(text, old, new):
    pattern = r"\b" + re.escape(old) + r"\b"
    result = re.sub(pattern, new, text, count=1, flags=re.IGNORECASE)
    return normalize_spaces(result)

def remove_first_occurrence(text, target):
    pattern = r"\b" + re.escape(target) + r"\b"
    result = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
    return normalize_spaces(result)

#knowledge base extraction part
def get_redundant_patterns(knowledge_base):
    patterns = {}
    for pattern, data in knowledge_base.items():
        if data.get("type") not in ["redundant_word", "redundant_phrase"]:
            continue

        normalized = normalize_phrase(pattern)
        if not normalized:
            continue

        patterns[normalized] = {
            "score": get_score(data),
            "ngram_size": data.get("ngram_size", len(normalized.split())),
            "type": data.get("type")
        }

    return patterns


def get_single_word_score(word, knowledge_base):
    word = normalize_phrase(word)
    if word not in knowledge_base:
        return 0
    data = knowledge_base[word]
    if data.get("type") not in ["redundant_word", "redundant_phrase"]:
        return 0
    ngram_size = data.get("ngram_size", len(word.split()))
    if ngram_size != 1:
        return 0
    return get_score(data)


def phrase_redundancy_coverage(phrase, knowledge_base, threshold=3.5):
    words = normalize_phrase(phrase).split()

    if not words:
        return 0
    redundant_count = 0
    for word in words:
        if get_single_word_score(word, knowledge_base) >= threshold:
            redundant_count += 1
    return redundant_count / len(words)


def contains_direct_repetition(phrase):
    words = normalize_phrase(phrase).split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            return True
    return False

# optimizer logic pt gui
def detect_direct_repetitions(prompt):
    suggestions = []
    pattern = r"\b(\w+)\s+\1\b"

    for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
        repeated_phrase = match.group()
        repeated_word = match.group(1)

        preview = replace_first_occurrence(
            prompt,
            repeated_phrase,
            repeated_word
        )

        suggestions.append({
            "type": "direct_repetition",
            "target": repeated_phrase.lower(),
            "replacement": repeated_word,
            "score": 999,
            "saved_tokens": count_tokens(prompt) - count_tokens(preview),
            "ngram_size": 2,
            "reason": "Direct repeated word detected",
            "preview": preview
        })

    return suggestions


def detect_knowledge_matches(prompt, knowledge_base):
    suggestions = []
    seen = set()

    for phrase, data in knowledge_base.items():
        if data.get("type") not in ["redundant_word", "redundant_phrase"]:
            continue

        normalized = normalize_phrase(phrase)

        if not normalized or normalized in seen:
            continue

        if contains_direct_repetition(normalized):
            continue

        pattern = r"\b" + re.escape(normalized) + r"\b"

        if not re.search(pattern, prompt, flags=re.IGNORECASE):
            continue

        ngram_size = data.get("ngram_size", len(normalized.split()))

        if ngram_size > 1:
            coverage = phrase_redundancy_coverage(normalized, knowledge_base)

            if coverage < 0.6:
                continue

        preview = remove_first_occurrence(prompt, normalized)
        saved_tokens = count_tokens(prompt) - count_tokens(preview)

        suggestions.append({
            "type": "redundant_phrase",
            "target": normalized,
            "original_phrase": normalized,
            "score": get_score(data),
            "saved_tokens": saved_tokens,
            "ngram_size": ngram_size,
            "reason": f"Knowledge base match: {ngram_size}-gram",
            "preview": preview
        })

        seen.add(normalized)

    return suggestions


def sort_suggestions(suggestions):
    def priority(suggestion):
        type_priority = 2 if suggestion["type"] == "direct_repetition" else 1

        return (
            type_priority,
            suggestion.get("ngram_size", 1),
            suggestion.get("score", 0),
            suggestion.get("saved_tokens", 0),
            len(suggestion.get("target", ""))
        )

    return sorted(suggestions, key=priority, reverse=True)


def generate_suggestions(prompt, knowledge_base, ignored_targets):
    suggestions = []

    suggestions.extend(detect_direct_repetitions(prompt))
    suggestions.extend(detect_knowledge_matches(prompt, knowledge_base))

    suggestions = [
        suggestion
        for suggestion in suggestions
        if suggestion["target"] not in ignored_targets
    ]

    return sort_suggestions(suggestions)


def apply_suggestion(prompt, suggestion):
    if suggestion["type"] == "direct_repetition":
        return replace_first_occurrence(
            prompt,
            suggestion["target"],
            suggestion["replacement"]
        )
    if suggestion["type"] == "redundant_phrase":
        return remove_first_occurrence(
            prompt,
            suggestion["target"]
        )
    return prompt

# THREAT DETECTION
def describe_unicode_char(char):
    #Returns useful information about a Unicode character.
    codepoint = f"U+{ord(char):04X}"

    try:
        name = unicodedata.name(char)
    except ValueError:
        name = "UNKNOWN CHARACTER"

    category = unicodedata.category(char)
    return codepoint, name, category


def is_suspicious_unicode_char(char):
    #Detects Unicode characters that may be suspicious in prompts.
    code = ord(char)
    category = unicodedata.category(char)

    # Normal ASCII is not suspicious
    if code < 128:
        return False
    # Format characters, including many zero-width characters
    if category == "Cf":
        return True
    if category in ["Cc", "Cs", "Co", "Cn"]:
        return True
    if char == "\uFFFD":
        return True
    name = unicodedata.name(char, "")
    if "RIGHT-TO-LEFT" in name or "LEFT-TO-RIGHT" in name:
        return True
    if "ZERO WIDTH" in name:
        return True
    if "CYRILLIC" in name or "GREEK" in name:
        return True
    return False


def detect_unicode_threats(prompt):
    #Detects suspicious Unicode characters directly from the prompt.
    threats = []

    for index, char in enumerate(prompt):
        if not is_suspicious_unicode_char(char):
            continue

        codepoint, name, category = describe_unicode_char(char)

        if unicodedata.category(char) == "Cf":
            threat_type = "Hidden / Zero-width Unicode"
            severity = "HIGH"
        elif "CYRILLIC" in name or "GREEK" in name:
            threat_type = "Unicode Spoofing / Homoglyph"
            severity = "MEDIUM"
        else:
            threat_type = "Suspicious Unicode Character"
            severity = "MEDIUM"

        threats.append({
            "type": threat_type,
            "severity": severity,
            "match": char,
            "display": f"{codepoint} {name} ({category})",
            "start": index,
            "end": index + 1
        })
    return threats


def fallback_detect_threats(prompt):
    #detecteaza threaturi locale bazate pe regex
    threat_rules = [
        {
            "type": "Prompt Injection",
            "severity": "HIGH",
            "pattern": r"ignore\s+(all\s+)?(previous|prior)\s+instructions"
        },
        {
            "type": "System Prompt Extraction",
            "severity": "HIGH",
            "pattern": r"reveal\s+the\s+system\s+prompt"
        },
        {
            "type": "System Prompt Extraction",
            "severity": "HIGH",
            "pattern": r"(show|print|leak|expose)\s+(the\s+)?(system\s+prompt|hidden\s+instructions)"
        },
        {
            "type": "Role Manipulation",
            "severity": "HIGH",
            "pattern": r"(act\s+as|pretend\s+to\s+be|you\s+are\s+now)\s+(a\s+)?(developer|admin|system)"
        },
        {
            "type": "Jailbreak Attempt",
            "severity": "HIGH",
            "pattern": r"(jailbreak|bypass|do\s+anything\s+now|dan\s+mode)"
        },
        {
            "type": "Suspicious Encoding",
            "severity": "MEDIUM",
            "pattern": r"(base64|unicode|zero\s+width)"
        },
        {
            "type": "Role Manipulation",
            "severity": "HIGH",
            "pattern": r"pretend\s+to\s+be\s+(a\s+)?(hacker|attacker|criminal|admin|developer|system)"
        },
        {
            "type": "Safety Bypass Attempt",
            "severity": "HIGH",
            "pattern": r"bypass\s+(all\s+)?(safety\s+rules|safety|rules|restrictions|filters)"
        }
    ]

    threats = []

    for rule in threat_rules:
        for match in re.finditer(rule["pattern"], prompt, flags=re.IGNORECASE):
            threats.append({
                "type": rule["type"],
                "severity": rule["severity"],
                "match": match.group(),
                "start": match.start(),
                "end": match.end()
            })

    hidden_chars = ["\u200b", "\u200c", "\u200d", "\ufeff"]
    threats.extend(detect_unicode_threats(prompt))
    return threats


def normalize_threat_item(item):
    #Converts different threat formats into the format expected by the GUI
    if isinstance(item, str):
        return {
            "type": "Threat detected",
            "severity": "MEDIUM",
            "match": item
        }

    if not isinstance(item, dict):
        return {
            "type": "Threat detected",
            "severity": "MEDIUM",
            "match": str(item)
        }

    threat_type = (
        item.get("type")
        or item.get("threat_type")
        or item.get("category")
        or item.get("name")
        or item.get("label")
        or "Threat detected"
    )

    severity = (
        item.get("severity")
        or item.get("level")
        or item.get("risk_level")
        or item.get("risk")
        or "MEDIUM"
    )

    match = (
        item.get("match")
        or item.get("matched_text")
        or item.get("pattern")
        or item.get("keyword")
        or item.get("text")
        or item.get("phrase")
        or item.get("detected")
        or item.get("message")
        or ""
    )
    return {
        "type": threat_type,
        "severity": str(severity).upper(),
        "match": str(match)
    }

def detect_threats(prompt):
    threats = []

    if external_threats is not None:
        possible_function_names = [
            "detect_threats",
            "analyze_threats",
            "scan_prompt",
            "scan_threats",
            "detect_prompt_threats"
        ]

        for function_name in possible_function_names:
            function = getattr(external_threats, function_name, None)

            if callable(function):
                try:
                    result = function(prompt)

                    if isinstance(result, dict):
                        if "threats" in result and isinstance(result["threats"], list):
                            threats.extend(
                                normalize_threat_item(item)
                                for item in result["threats"]
                            )
                        else:
                            threats.append(normalize_threat_item(result))

                    elif isinstance(result, list):
                        threats.extend(
                            normalize_threat_item(item)
                            for item in result
                        )

                    elif result:
                        threats.append(normalize_threat_item(result))

                except Exception as error:
                    threats.append({
                        "type": "Threat module error",
                        "severity": "LOW",
                        "match": str(error)
                    })

                break

    fallback_threats = fallback_detect_threats(prompt)
    threats.extend(fallback_threats)
    cleaned = []

    for threat in threats:
        match = threat.get("match", "")
        if not match:
            continue
        cleaned.append(threat)

    unique = []
    seen = set()

    for threat in cleaned:
        key = (
            threat.get("type"),
            threat.get("severity"),
            threat.get("match")
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(threat)

    return unique

# GUI APPLICATION
class TokenizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tokenization Optimizer and Threat Detection")
        self.root.geometry("1250x820")

        self.knowledge_base = load_knowledge_base(KNOWLEDGE_FILE)
        self.redundant_patterns = get_redundant_patterns(self.knowledge_base)

        self.optimized_prompt = ""
        self.current_suggestion = None
        self.ignored_targets = set()
        self.accepted_changes = []

        self.create_widgets()

    def create_widgets(self):
        title = tk.Label(
            self.root,
            text="Tokenization, Redundancy & Threat Analyzer",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=8)

        subtitle = tk.Label(
            self.root,
            text="Analyze LLM prompts at token level, highlight redundancy, detect threats, and optimize interactively.",
            font=("Arial", 10)
        )
        subtitle.pack(pady=2)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=12, pady=8)

        top_frame = tk.Frame(main_frame)
        top_frame.pack(fill="both", expand=False)

        original_frame = tk.LabelFrame(top_frame, text="Original Prompt")
        original_frame.pack(side="left", fill="both", expand=True, padx=5)

        optimized_frame = tk.LabelFrame(top_frame, text="Optimized Prompt Preview")
        optimized_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.prompt_text = tk.Text(
            original_frame,
            height=9,
            wrap="word",
            font=("Arial", 11)
        )
        self.prompt_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.optimized_text = tk.Text(
            optimized_frame,
            height=9,
            wrap="word",
            font=("Arial", 11)
        )
        self.optimized_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.configure_tags()

        sample_prompt = (
            "Could you please please summarize this article in a very very "
            "short and concise way for me please?"
        )
        self.prompt_text.insert("1.0", sample_prompt)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill="x", pady=8)

        tk.Button(
            button_frame,
            text="Analyze",
            width=18,
            bg="#d9ead3",
            command=self.analyze_prompt
        ).pack(side="left", padx=4)

        tk.Button(
            button_frame,
            text="Suggest Optimization",
            width=20,
            bg="#cfe2f3",
            command=self.show_next_suggestion
        ).pack(side="left", padx=4)

        tk.Button(
            button_frame,
            text="Accept",
            width=12,
            bg="#b6d7a8",
            command=self.accept_suggestion
        ).pack(side="left", padx=4)

        tk.Button(
            button_frame,
            text="Reject",
            width=12,
            bg="#f9cb9c",
            command=self.reject_suggestion
        ).pack(side="left", padx=4)

        tk.Button(
            button_frame,
            text="Load File",
            width=12,
            command=self.load_prompt_file
        ).pack(side="left", padx=4)

        tk.Button(
            button_frame,
            text="Save Optimized",
            width=16,
            command=self.save_optimized_prompt
        ).pack(side="left", padx=4)

        tk.Button(
            button_frame,
            text="Clear",
            width=12,
            bg="#f4cccc",
            command=self.clear_all
        ).pack(side="left", padx=4)

        middle_frame = tk.Frame(main_frame)
        middle_frame.pack(fill="both", expand=True)

        stats_frame = tk.LabelFrame(middle_frame, text="Token Statistics")
        stats_frame.pack(side="left", fill="both", expand=True, padx=5)

        suggestion_frame = tk.LabelFrame(middle_frame, text="Current Optimization Suggestion")
        suggestion_frame.pack(side="left", fill="both", expand=True, padx=5)

        threats_frame = tk.LabelFrame(middle_frame, text="Threat Detection")
        threats_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.stats_text = tk.Text(
            stats_frame,
            height=10,
            wrap="word",
            font=("Consolas", 10)
        )
        self.stats_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.suggestion_text = tk.Text(
            suggestion_frame,
            height=10,
            wrap="word",
            font=("Consolas", 10)
        )
        self.suggestion_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.threats_text = tk.Text(
            threats_frame,
            height=10,
            wrap="word",
            font=("Consolas", 10)
        )
        self.threats_text.pack(fill="both", expand=True, padx=5, pady=5)

        token_frame = tk.LabelFrame(main_frame, text="Tokenization Output")
        token_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.tokens_text = tk.Text(
            token_frame,
            height=10,
            wrap="word",
            font=("Consolas", 10)
        )
        self.tokens_text.pack(fill="both", expand=True, padx=5, pady=5)

        report_frame = tk.LabelFrame(main_frame, text="Redundancy Report")
        report_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.report_text = tk.Text(
            report_frame,
            height=8,
            wrap="word",
            font=("Consolas", 10)
        )
        self.report_text.pack(fill="both", expand=True, padx=5, pady=5)

        progress_frame = tk.Frame(main_frame)
        progress_frame.pack(fill="x", pady=5)

        self.reduction_label = tk.Label(
            progress_frame,
            text="Token reduction: 0%",
            font=("Arial", 10, "bold")
        )
        self.reduction_label.pack(side="left", padx=5)

        self.reduction_bar = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=300,
            mode="determinate"
        )
        self.reduction_bar.pack(side="left", padx=5)

    def configure_tags(self):
        self.prompt_text.tag_configure(
            "redundant_word",
            foreground="#cc6600",
            underline=True,
            font=("Arial", 11, "bold")
        )

        self.prompt_text.tag_configure(
            "direct_repetition",
            foreground="white",
            background="#cc0000",
            font=("Arial", 11, "bold")
        )

        self.prompt_text.tag_configure(
            "threat",
            foreground="white",
            background="#6a0dad",
            font=("Arial", 11, "bold")
        )
        self.prompt_text.tag_configure(
            "unicode_threat",
            foreground="black",
            background="#ffeb3b",
            font=("Arial", 11, "bold")
        )

        self.prompt_text.tag_raise("unicode_threat")
        self.prompt_text.tag_raise("threat")
        self.prompt_text.tag_raise("direct_repetition")

    # ANALYSIS
    def analyze_prompt(self):
        prompt = self.prompt_text.get("1.0", "end-1c")

        if not prompt.strip():
            messagebox.showwarning("Warning", "Please enter a prompt first.")
            return

        self.optimized_prompt = normalize_spaces(prompt)
        self.current_suggestion = None
        self.ignored_targets = set()
        self.accepted_changes = []

        self.optimized_text.delete("1.0", tk.END)
        self.optimized_text.insert("1.0", self.optimized_prompt)

        self.clear_analysis_outputs()

        token_ids, decoded_tokens = get_tokens(prompt)
        word_count = len(re.findall(r"\b\w+\b", prompt))
        token_count_value = len(token_ids)
        tokens_per_word = token_count_value / max(word_count, 1)

        direct_repetitions = self.highlight_direct_repetitions(prompt)
        redundant_patterns = self.highlight_redundant_words_only(prompt)
        threats = self.highlight_threats(prompt)

        self.show_statistics(
            token_count_value,
            word_count,
            tokens_per_word,
            redundant_patterns,
            direct_repetitions,
            threats
        )

        self.show_tokens(token_ids, decoded_tokens)
        self.show_redundancy_report(redundant_patterns, direct_repetitions)
        self.show_threat_report(threats)
        self.update_reduction_stats()

    def clear_analysis_outputs(self):
        self.stats_text.delete("1.0", tk.END)
        self.tokens_text.delete("1.0", tk.END)
        self.report_text.delete("1.0", tk.END)
        self.threats_text.delete("1.0", tk.END)
        self.suggestion_text.delete("1.0", tk.END)

        self.prompt_text.tag_remove("redundant_word", "1.0", tk.END)
        self.prompt_text.tag_remove("redundant_phrase", "1.0", tk.END)
        self.prompt_text.tag_remove("direct_repetition", "1.0", tk.END)
        self.prompt_text.tag_remove("threat", "1.0", tk.END)
        self.prompt_text.tag_remove("unicode_threat", "1.0", tk.END)

    def highlight_direct_repetitions(self, prompt):
        found = []
        pattern = r"\b(\w+)\s+\1\b"

        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            start_index = f"1.0 + {match.start()} chars"
            end_index = f"1.0 + {match.end()} chars"

            self.prompt_text.tag_add(
                "direct_repetition",
                start_index,
                end_index
            )

            found.append({
                "pattern": match.group(),
                "type": "direct_repetition",
                "ngram_size": 2
            })

        return found

    def highlight_redundant_words_only(self, prompt):
        #Highlights only strong single-word redundancy
        found = []
        # Higher threshold = less visual noise
        min_visual_score = 7.0
        repeated_ranges = []
        repetition_pattern = r"\b(\w+)\s+\1\b"
        for match in re.finditer(repetition_pattern, prompt, flags=re.IGNORECASE):
            repeated_ranges.append((match.start(), match.end()))

        words = re.finditer(r"\b\w+\b", prompt)

        for match in words:
            start = match.start()
            end = match.end()

            inside_repetition = False

            for repeated_start, repeated_end in repeated_ranges:
                if start >= repeated_start and end <= repeated_end:
                    inside_repetition = True
                    break

            if inside_repetition:
                continue

            word = match.group()
            normalized = normalize_phrase(word)

            if normalized not in self.redundant_patterns:
                continue

            data = self.redundant_patterns[normalized]

            if data["ngram_size"] != 1:
                continue

            if data["score"] < min_visual_score:
                continue

            start_index = f"1.0 + {match.start()} chars"
            end_index = f"1.0 + {match.end()} chars"

            self.prompt_text.tag_add(
                "redundant_word",
                start_index,
                end_index
            )

            found.append({
                "pattern": word,
                "score": data["score"],
                "ngram_size": data["ngram_size"],
                "type": data["type"]
            })

        return found

    def highlight_threats(self, prompt):
        threats = detect_threats(prompt)

        for threat in threats:
            tag_name = (
                "unicode_threat"
                if "unicode" in threat.get("type", "").lower()
                or "homoglyph" in threat.get("type", "").lower()
                else "threat"
            )

            if "start" in threat and "end" in threat:
                start_index = f"1.0 + {threat['start']} chars"
                end_index = f"1.0 + {threat['end']} chars"

                self.prompt_text.tag_add(
                    tag_name,
                    start_index,
                    end_index
                )

                continue

            match_text = str(threat.get("match", ""))

            if not match_text:
                continue

            regex_pattern = re.escape(match_text)

            for match in re.finditer(regex_pattern, prompt, flags=re.IGNORECASE):
                start_index = f"1.0 + {match.start()} chars"
                end_index = f"1.0 + {match.end()} chars"

                self.prompt_text.tag_add(
                    tag_name,
                    start_index,
                    end_index
                )

        self.prompt_text.tag_raise("unicode_threat")
        self.prompt_text.tag_raise("threat")
        self.prompt_text.tag_raise("direct_repetition")

        return threats

   
    # OPTIMIZATION BUTTONS
    def show_next_suggestion(self):
        if not self.optimized_prompt:
            self.optimized_prompt = normalize_spaces(
                self.prompt_text.get("1.0", "end-1c")
            )

        suggestions = generate_suggestions(
            self.optimized_prompt,
            self.knowledge_base,
            self.ignored_targets
        )

        self.suggestion_text.delete("1.0", tk.END)

        if not suggestions:
            self.current_suggestion = None
            self.suggestion_text.insert(
                tk.END,
                "No more suggestions found.\n"
            )
            return

        self.current_suggestion = suggestions[0]
        self.display_current_suggestion(self.current_suggestion)

    def display_current_suggestion(self, suggestion):
        if suggestion["type"] == "direct_repetition":
            action = (
                f"Replace: '{suggestion['target']}' "
                f"-> '{suggestion['replacement']}'"
            )
        else:
            action = f"Remove: '{suggestion['original_phrase']}'"

        text = ""
        text += f"Type: {suggestion['type']}\n"
        text += f"Action: {action}\n"
        text += f"N-gram size: {suggestion['ngram_size']}\n"
        text += f"Score: {suggestion['score']}\n"
        text += f"Estimated saved tokens: {suggestion['saved_tokens']}\n"
        text += f"Reason: {suggestion['reason']}\n\n"
        text += "Preview:\n"
        text += suggestion["preview"]

        self.suggestion_text.insert(tk.END, text)

    def accept_suggestion(self):
        if self.current_suggestion is None:
            messagebox.showinfo(
                "Info",
                "No active suggestion. Click 'Suggest Optimization' first."
            )
            return

        self.optimized_prompt = apply_suggestion(
            self.optimized_prompt,
            self.current_suggestion
        )
        self.optimized_prompt = normalize_spaces(self.optimized_prompt)

        self.accepted_changes.append(self.current_suggestion)

        self.optimized_text.delete("1.0", tk.END)
        self.optimized_text.insert("1.0", self.optimized_prompt)

        self.current_suggestion = None
        self.suggestion_text.delete("1.0", tk.END)

        self.update_reduction_stats()
        self.show_next_suggestion()

    def reject_suggestion(self):
        if self.current_suggestion is None:
            messagebox.showinfo(
                "Info",
                "No active suggestion. Click 'Suggest Optimization' first."
            )
            return

        self.ignored_targets.add(self.current_suggestion["target"])

        self.current_suggestion = None
        self.suggestion_text.delete("1.0", tk.END)

        self.show_next_suggestion()

    # OUTPUT PANELS
    def show_statistics(
        self,
        token_count_value,
        word_count,
        tokens_per_word,
        redundant_patterns,
        direct_repetitions,
        threats
    ):
        stats = ""
        stats += f"Token count:        {token_count_value}\n"
        stats += f"Word count:         {word_count}\n"
        stats += f"Tokens per word:    {tokens_per_word:.2f}\n"
        stats += f"Redundant words:    {len(redundant_patterns)}\n"
        stats += f"Direct repetitions: {len(direct_repetitions)}\n"
        stats += f"Threats detected:   {len(threats)}\n"

        if token_count_value <= 20:
            efficiency = "Good"
        elif token_count_value <= 50:
            efficiency = "Medium"
        else:
            efficiency = "High token usage"

        stats += f"Efficiency level:   {efficiency}\n"

        self.stats_text.insert(tk.END, stats)

    def show_tokens(self, token_ids, decoded_tokens):
        for index, (token_id, token_text) in enumerate(
            zip(token_ids, decoded_tokens),
            start=1
        ):
            safe_token_text = token_text.replace("\n", "\\n")

            self.tokens_text.insert(
                tk.END,
                f"{index}. ID={token_id} | text='{safe_token_text}'\n"
            )

    def show_redundancy_report(self, redundant_patterns, direct_repetitions):
        if not redundant_patterns and not direct_repetitions:
            self.report_text.insert(
                tk.END,
                "No redundant words or repetitions detected.\n"
            )
            return

        if direct_repetitions:
            self.report_text.insert(
                tk.END,
                "Direct repetitions detected:\n"
            )

            for item in direct_repetitions:
                self.report_text.insert(
                    tk.END,
                    f"- '{item['pattern']}' | repeated adjacent words\n"
                )

            self.report_text.insert(tk.END, "\n")

        if redundant_patterns:
            self.report_text.insert(
                tk.END,
                "Redundant words from knowledge base:\n"
            )

            for item in redundant_patterns:
                self.report_text.insert(
                    tk.END,
                    f"- '{item['pattern']}' | "
                    f"score: {item['score']} | "
                    f"type: {item['type']}\n"
                )

    def show_threat_report(self, threats):
        if not threats:
            self.threats_text.insert(
                tk.END,
                "No threats detected.\n"
            )
            return

        for threat in threats:
            threat_type = threat.get("type", "Threat detected")
            severity = threat.get("severity", "MEDIUM")

            display = threat.get("display")
            match = threat.get("match", "")

            if display:
                shown_match = display
            elif match:
                shown_match = match
            else:
                shown_match = "-"

            self.threats_text.insert(
                tk.END,
                f"- {threat_type} | severity: {severity} | match: {shown_match}\n"
            )

    def update_reduction_stats(self):
        original_prompt = self.prompt_text.get("1.0", "end-1c")
        optimized_prompt = self.optimized_text.get("1.0", "end-1c")

        original_tokens = count_tokens(original_prompt)
        optimized_tokens = count_tokens(optimized_prompt)

        if original_tokens <= 0:
            reduction = 0
        else:
            reduction = ((original_tokens - optimized_tokens) / original_tokens) * 100

        reduction = max(0, min(100, reduction))

        self.reduction_label.config(
            text=f"Token reduction: {reduction:.2f}%"
        )
        self.reduction_bar["value"] = reduction

    # FILE / CLEAR ACTIONS
    def load_prompt_file(self):
        path = filedialog.askopenfilename(
            initialdir=BASE_DIR,
            title="Select prompt file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not path:
            return

        with open(path, "r", encoding="utf-8") as file:
            prompt = file.read()

        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", prompt)

        self.optimized_text.delete("1.0", tk.END)
        self.optimized_prompt = ""
        self.clear_analysis_outputs()

    def save_optimized_prompt(self):
        optimized_prompt = self.optimized_text.get("1.0", "end-1c")

        if not optimized_prompt.strip():
            messagebox.showwarning(
                "Warning",
                "There is no optimized prompt to save."
            )
            return

        path = filedialog.asksaveasfilename(
            initialdir=BASE_DIR,
            title="Save optimized prompt",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not path:
            return

        with open(path, "w", encoding="utf-8") as file:
            file.write(optimized_prompt)

        messagebox.showinfo(
            "Saved",
            "Optimized prompt saved successfully."
        )

    def clear_all(self):
        self.prompt_text.delete("1.0", tk.END)
        self.optimized_text.delete("1.0", tk.END)

        self.optimized_prompt = ""
        self.current_suggestion = None
        self.ignored_targets = set()
        self.accepted_changes = []

        self.clear_analysis_outputs()
        self.reduction_label.config(text="Token reduction: 0%")
        self.reduction_bar["value"] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = TokenizerApp(root)
    root.mainloop()