# TODO: use tiktoken to process text
# feed training data -> json knowledge dictionary 

import json
import re 
from collections import Counter
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.readlines()
    
# set all text to lowercase, remove extra spaces and whitespaces
def normalize_text(text):
    text = text.lower()

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# use regex to extract all words
def extract_words(text):
    return re.findall(r"\b\w+\b", text.lower())

# compute cost
def token_count(text):
    return len(encoding.encode(text))

# word frequency analysis 
def build_frequency_map(lines):
    counter = Counter()

    for line in lines:
        normalized = normalize_text(line)

        words = extract_words(normalized)

        counter.update(words)

    return counter

# repetition detection
def detect_repeated_phrases(lines):
    repeated = Counter()

    for line in lines:
        words = extract_words(line)

        for i in range(len(words) - 1):

            if words[i] == words[i + 1]:

                phrase = f"{words[i]} {words[i + 1]}"

                repeated[phrase] += 1

    return repeated

# computes redundancy scores for words based on their frequency in the normal text
# vs. their frequency in the repeated phrases

def compute_redundancy_scores(
    normal_freq,
    redundant_freq
):
    scores = {}

    for word, redundant_count in redundant_freq.items():

        normal_count = normal_freq.get(word, 1)

        score = redundant_count / normal_count

        # threshold
        if score >= 3:

            scores[word] = {
                "redundancy_score": round(score, 2),
                "normal_frequency": normal_count,
                "redundant_frequency": redundant_count
            }

    return scores


def build_knowledge_base(redundancy_scores, repeated_phrases):
    knowledge = {}

    for word, data in redundancy_scores.items():

        knowledge[word] = {
            "type": "redundant_word",
            "score": data["redundancy_score"],
            "normal_frequency": data["normal_frequency"],
            "redundant_frequency": data["redundant_frequency"],
            "suggested_action": "remove_or_replace"
        }

    for phrase, freq in repeated_phrases.items():

        knowledge[phrase] = {
            "type": "repeated_phrase",
            "frequency": freq,
            "suggested_action": "compress"
        }

    return knowledge

def save_knowledge_base(data, path):

    with open(path, "w", encoding="utf-8") as file:

        json.dump(
            data,
            file,
            indent=4
        )


def analyze_prompt(prompt):

    tokens = token_count(prompt)

    words = extract_words(prompt)

    return {
        "tokens": tokens,
        "word_count": len(words),
        "tokens_per_word": round(tokens / max(len(words), 1), 2)
    }


def main():

    print("Loading datasets...")

    normal_lines = load_dataset("data/normal_prompts.txt")

    redundant_lines = load_dataset("data/redundant_prompts.txt")

    print("Building frequency maps...")

    normal_freq = build_frequency_map(normal_lines)

    redundant_freq = build_frequency_map(redundant_lines)
    print(normal_freq.most_common(20))
    print(redundant_freq.most_common(20))

    print("Detecting repeated phrases...")

    repeated_phrases = detect_repeated_phrases(redundant_lines)

    print("Computing redundancy scores...")

    redundancy_scores = compute_redundancy_scores(normal_freq, redundant_freq)

    print("Building knowledge base...")

    knowledge = build_knowledge_base(redundancy_scores, repeated_phrases)

    save_knowledge_base(knowledge, "C:\\Users\\User\\Desktop\\MP\\MP1_Tokenizare\\knowledge\\redundant_patterns.json")

    print("\nKnowledge base generated.")

    print(f"Patterns found: {len(knowledge)}")

    # sample test
    sample = redundant_lines[0]

    print("\nSample Prompt:")
    print(sample)

    stats = analyze_prompt(sample)

    print("\nPrompt Statistics:")
    print(stats)



if __name__ == "__main__":
    main()