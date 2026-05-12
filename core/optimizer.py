import json
import re
import tiktoken


encoding = tiktoken.get_encoding("cl100k_base")


def load_knowledge_base(path):
    #Loads the redundancy knowledge base 
    
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_prompt_from_file(path):
    #Reads a prompt from a text file
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def save_prompt_to_file(path, prompt):
    #Saves the optimized prompt back in the file
    with open(path, "w", encoding="utf-8") as file:
        file.write(prompt)


def count_tokens(text):
    return len(encoding.encode(text))


def normalize_spaces(text):
    #Removes duplicated spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_word(word):
    #Removes punctuation and converts a word to lowercase
    return re.sub(r"[^\w]", "", word).lower()


def replace_first_occurrence(text, old, new):
    pattern = r"\b" + re.escape(old) + r"\b"
    return re.sub(pattern, new, text, count=1, flags=re.IGNORECASE)


def remove_first_word_occurrence(text, target_word):
    words = text.split()
    result = []
    removed = False

    for word in words:
        if not removed and clean_word(word) == target_word:
            removed = True
            continue

        result.append(word)

    return normalize_spaces(" ".join(result))


def detect_repeated_phrase_suggestions(prompt, knowledge_base):
    #Detects repeated phrases from the knowledge base
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


def detect_redundant_word_suggestions(prompt, knowledge_base, min_score=4.0):
    #Detects suggestions words to be removed if the users allows

    suggestions = []
    already_suggested = set()

    words = prompt.split()

    for word in words:
        cleaned = clean_word(word)

        if cleaned in already_suggested:
            continue

        if cleaned not in knowledge_base:
            continue

        data = knowledge_base[cleaned]

        if data.get("type") != "redundant_word":
            continue

        score = data.get("score", 0)

        if score < min_score:
            continue

        preview = remove_first_word_occurrence(prompt, cleaned)

        original_tokens = count_tokens(prompt)
        preview_tokens = count_tokens(preview)
        saved_tokens = original_tokens - preview_tokens

        if saved_tokens <= 0:
            continue

        suggestions.append({
            "type": "redundant_word",
            "target": cleaned,
            "original_word": word,
            "score": score,
            "saved_tokens": saved_tokens,
            "reason": f"High redundancy score: {score}",
            "preview": preview
        })

        already_suggested.add(cleaned)

    return suggestions


def generate_suggestions(prompt, knowledge_base, min_score=4.0):
    #Generates all optimization suggestions
    suggestions = []

    repeated_suggestions = detect_repeated_phrase_suggestions(
        prompt,
        knowledge_base
    )

    word_suggestions = detect_redundant_word_suggestions(
        prompt,
        knowledge_base,
        min_score=min_score
    )

    suggestions.extend(repeated_suggestions)
    suggestions.extend(word_suggestions)

    return suggestions


def apply_suggestion(prompt, suggestion):
    #Applies suggestion one by one
    if suggestion["type"] == "repeated_phrase":
        return replace_first_occurrence(
            prompt,
            suggestion["target"],
            suggestion["replacement"]
        )

    if suggestion["type"] == "redundant_word":
        return remove_first_word_occurrence(
            prompt,
            suggestion["target"]
        )

    return prompt


def interactive_optimize_prompt(prompt, knowledge_base, min_score=4.0):
    #Shows suggestions one by one asking users to accept them or not
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

        elif suggestion["type"] == "redundant_word":
            print(f"Type: Redundant word")
            print(f"Remove: '{suggestion['original_word']}'")
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
                knowledge_base[target]["score"] = 0

        elif choice == "stop":
            break

        else:
            print("Invalid option. Please type y, n, or stop.")

    return current_prompt, accepted_changes


def show_final_report(original_prompt, optimized_prompt, accepted_changes):
    #Shows the final optimization report
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

            elif change["type"] == "redundant_word":
                print(
                    f"[Removed word] '{change['original_word']}' "
                    f"(score: {change['score']})"
                )