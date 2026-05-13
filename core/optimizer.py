import json
import re
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def load_knowledge_base(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def load_prompt_from_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def save_prompt_to_file(path, prompt):
    with open(path, "w", encoding="utf-8") as file:
        file.write(prompt)

def count_tokens(text):
    return len(encoding.encode(text))


def normalize_spaces(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    return text.strip()

def normalize_phrase(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

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
    #Checks how much of a phrase is made of individually redundant words

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

def detect_direct_repetitions(prompt):
    #Detects repeated words directly from prompt

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
    #Finds all words and phrases from knowledge base inside the prompt

    suggestions = []
    seen = set()

    for phrase, data in knowledge_base.items():
        if data.get("type") not in ["redundant_word", "redundant_phrase"]:
            continue

        normalized = normalize_phrase(phrase)

        if not normalized or normalized in seen:
            continue

        # Repeated structures are handled separately as direct repetitions
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
    ### Priority: 1 Direct repetitions first 2 Larger n-grams 3 Higher score
    
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


def print_suggestion(step, current_prompt, suggestion):
    print(f"\n STEP {step}")

    print("\nCURRENT PROMPT")
    print(current_prompt)

    print("\nSUGGESTION")

    if suggestion["type"] == "direct_repetition":
        print("Type: Direct repetition")
        print(f"Replace: '{suggestion['target']}' -> '{suggestion['replacement']}'")

    else:
        print("Type: Knowledge base word/phrase")
        print(f"Remove: '{suggestion['original_phrase']}'")
        print(f"Score: {suggestion['score']}")

        if suggestion["score"] < 4:
            print("Warning: low-score suggestion. Review carefully.")

    print(f"N-gram size: {suggestion['ngram_size']}")
    print(f"Estimated saved tokens: {suggestion['saved_tokens']}")
    print(f"Reason: {suggestion['reason']}")

    print("\nPREVIEW")
    print(suggestion["preview"])

def recursive_optimize(
    current_prompt,
    knowledge_base,
    accepted_changes,
    ignored_targets,
    step
):
    suggestions = generate_suggestions(
        current_prompt,
        knowledge_base,
        ignored_targets
    )

    if not suggestions:
        print("\nNO MORE CURRENT SUGGESTIONS")
        print("Type 'stop' to finish")

        choice = input("Choice: ").strip().lower()

        if choice == "stop":
            return current_prompt, accepted_changes

        return recursive_optimize(
            current_prompt,
            knowledge_base,
            accepted_changes,
            ignored_targets,
            step + 1
        )

    suggestion = suggestions[0]

    print_suggestion(step, current_prompt, suggestion)

    choice = input("\nAccept this change? Type y/n/stop: ").strip().lower()

    if choice == "stop":
        return current_prompt, accepted_changes

    if choice == "y":
        new_prompt = apply_suggestion(current_prompt, suggestion)
        new_prompt = normalize_spaces(new_prompt)

        accepted_changes.append(suggestion)

        return recursive_optimize(
            new_prompt,
            knowledge_base,
            accepted_changes,
            ignored_targets,
            step + 1
        )

    if choice == "n":
        ignored_targets.add(suggestion["target"])

        return recursive_optimize(
            current_prompt,
            knowledge_base,
            accepted_changes,
            ignored_targets,
            step + 1
        )

    print("Invalid option. Please type y, n, or stop.")

    return recursive_optimize(
        current_prompt,
        knowledge_base,
        accepted_changes,
        ignored_targets,
        step
    )


def interactive_optimize_prompt(prompt, knowledge_base):
    return recursive_optimize(
        current_prompt=normalize_spaces(prompt),
        knowledge_base=knowledge_base,
        accepted_changes=[],
        ignored_targets=set(),
        step=1
    )

def show_final_report(original_prompt, optimized_prompt, accepted_changes):
    original_tokens = count_tokens(original_prompt)
    optimized_tokens = count_tokens(optimized_prompt)
    saved_tokens = original_tokens - optimized_tokens

    print("\nFINAL RESULT")
    print(optimized_prompt)

    print("\nTOKEN STATISTICS")
    print(f"Original tokens:  {original_tokens}")
    print(f"Optimized tokens: {optimized_tokens}")
    print(f"Saved tokens:     {saved_tokens}")

    if original_tokens > 0:
        reduction = (saved_tokens / original_tokens) * 100
        print(f"Reduction:        {reduction:.2f}%")

    print("\nACCEPTED CHANGES")

    if not accepted_changes:
        print("No changes were accepted.")
        return

    for change in accepted_changes:
        if change["type"] == "direct_repetition":
            print(
                f"[Repeated] '{change['target']}' "
                f"-> '{change['replacement']}'"
            )

        elif change["type"] == "redundant_phrase":
            print(
                f"[Removed] '{change['original_phrase']}' "
                f"(ngram: {change['ngram_size']}, score: {change['score']})"
            )