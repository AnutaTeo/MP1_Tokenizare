import json
import re
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")


def load_knowledge_base(path):
    # Loads the redundancy knowledge base 
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_prompt_from_file(path):
    # Reads a prompt from a text file
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def save_prompt_to_file(path, prompt):
    # Saves the optimized prompt back in the file
    with open(path, "w", encoding="utf-8") as file:
        file.write(prompt)


def count_tokens(text):
    return len(encoding.encode(text))


def normalize_spaces(text):
    # Removes duplicated spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_word(word):
    # Removes punctuation and converts a word to lowercase
    return re.sub(r"[^\w\s]", "", word).lower()


def replace_first_occurrence(text, old, new):
    pattern = r"\b" + re.escape(old) + r"\b"
    return re.sub(pattern, new, text, count=1, flags=re.IGNORECASE)


def remove_first_occurrence(text, target):
    # Uses regex to remove multi-word phrases or single words cleanly
    pattern = r"\b" + re.escape(target) + r"\b"
    result = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
    return normalize_spaces(result)


def detect_repeated_phrase_suggestions(prompt, knowledge_base):
    # Detects repeated phrases from the knowledge base
    suggestions = []

    for pattern, data in knowledge_base.items():
        if data.get("type") != "repeated_phrase":
            continue

        words = pattern.split()

        if len(words) == 2 and words[0] == words[1]:
            replacement = words[0]

            regex_pattern = r"\b" + re.escape(pattern) + r"\b"

            if re.search(regex_pattern, prompt, flags=re.IGNORECASE):
                preview = replace_first_occurrence(prompt, pattern, replacement)

                suggestions.append({
                    "type": "repeated_phrase",
                    "target": pattern,
                    "replacement": replacement,
                    "reason": "Repeated phrase detected",
                    "preview": preview
                })

    return suggestions


def detect_redundant_phrase_suggestions(prompt, knowledge_base, min_score=4.0):
    # Detects suggested words or phrases to be removed
    suggestions = []
    already_suggested = set()

    for phrase, data in knowledge_base.items():
        # Support both the old type and the new JSON "redundant_phrase" type
        if data.get("type") not in ["redundant_word", "redundant_phrase"]:
            continue

        # Look for the new "redundancy_score" key, fallback to "score"
        score = data.get("redundancy_score", data.get("score", 0))

        if score < min_score:
            continue

        # Use regex boundaries to find the phrase in the prompt
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if not re.search(pattern, prompt, flags=re.IGNORECASE):
            continue

        if phrase in already_suggested:
            continue

        preview = remove_first_occurrence(prompt, phrase)

        original_tokens = count_tokens(prompt)
        preview_tokens = count_tokens(preview)
        saved_tokens = original_tokens - preview_tokens

        if saved_tokens <= 0:
            continue

        suggestions.append({
            "type": "redundant_phrase", 
            "target": phrase,
            "original_phrase": phrase,
            "score": score,
            "saved_tokens": saved_tokens,
            "reason": f"High redundancy score: {score}",
            "preview": preview
        })

        already_suggested.add(phrase)

    return suggestions


def generate_suggestions(prompt, knowledge_base, min_score=4.0):
    # Generates all optimization suggestions
    suggestions = []

    repeated_suggestions = detect_repeated_phrase_suggestions(
        prompt,
        knowledge_base
    )

    phrase_suggestions = detect_redundant_phrase_suggestions(
        prompt,
        knowledge_base,
        min_score=min_score
    )

    suggestions.extend(repeated_suggestions)
    suggestions.extend(phrase_suggestions)

    return suggestions


def apply_suggestion(prompt, suggestion):
    # Applies suggestion one by one
    if suggestion["type"] == "repeated_phrase":
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


def interactive_optimize_prompt(prompt, knowledge_base, min_score=4.0):
    # Shows suggestions one by one asking users to accept them or not
    current_prompt = prompt
    accepted_changes = []

    while True:
        suggestions = generate_suggestions(
            current_prompt,
            knowledge_base,
            min_score=min_score
        )

        if not suggestions:
            break

        suggestion = suggestions[0]

        print("\nCURRENT PROMPT")
        print(current_prompt)

        print("\nSUGGESTION")

        if suggestion["type"] == "repeated_phrase":
            print(f"Type: Repeated phrase")
            print(f"Replace: '{suggestion['target']}' -> '{suggestion['replacement']}'")
            print(f"Reason: {suggestion['reason']}")

        elif suggestion["type"] == "redundant_phrase":
            print(f"Type: Redundant phrase")
            print(f"Remove: '{suggestion['original_phrase']}'")
            print(f"Reason: {suggestion['reason']}")
            print(f"Estimated saved tokens: {suggestion['saved_tokens']}")

        print("\nPREVIEW")
        print(suggestion["preview"])

        choice = input("\nAccept this change? Type y/n/stop: ").strip().lower()

        if choice == "y":
            current_prompt = apply_suggestion(current_prompt, suggestion)
            current_prompt = normalize_spaces(current_prompt)
            accepted_changes.append(suggestion)

        elif choice == "n":
            target = suggestion["target"]
            if target in knowledge_base:
                # Nullify the correct key so it isn't suggested again in this loop
                if "redundancy_score" in knowledge_base[target]:
                    knowledge_base[target]["redundancy_score"] = 0
                else:
                    knowledge_base[target]["score"] = 0

        elif choice == "stop":
            break

        else:
            print("Invalid option. Please type y, n, or stop.")

    return current_prompt, accepted_changes


def show_final_report(original_prompt, optimized_prompt, accepted_changes):
    # Shows the final optimization report
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
    else:
        for change in accepted_changes:
            if change["type"] == "repeated_phrase":
                print(
                    f"[Repeated phrase] '{change['target']}' "
                    f"-> '{change['replacement']}'"
                )

            elif change["type"] == "redundant_phrase":
                print(
                    f"[Removed phrase] '{change['original_phrase']}' "
                    f"(score: {change['score']})"
                )