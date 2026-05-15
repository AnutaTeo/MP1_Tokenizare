import json
import math
import re
from collections import Counter
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

# DATA LOADING
def load_dataset(path):

    with open(path, "r", encoding="utf-8") as file:

        return [

            line.strip()

            for line in file.readlines()

            if line.strip()
        ]



# TEXT NORMALIZATION
def normalize_text(text):

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()



# WORD EXTRACTION
def extract_words(text, normalize=True):
    # Skip lowercasing if text is already normalized
    if normalize:
        text = text.lower()
    return re.findall(r"\b\w+\b", text)



# TOKENIZATION
def tokenize_text(text):

    return encoding.encode(text)


def token_count(text):

    return len(tokenize_text(text))



# NGRAM EXTRACTION
def extract_ngrams(words, n=6):
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]


# MULTI-NGRAM EXTRACTION (ALL SIZES IN ONE PASS)
def extract_all_ngrams(words, sizes=[1, 2, 3, 4, 5]):
    """Extract multiple n-gram sizes in a single pass."""
    all_ngrams = []
    for n in sizes:
        all_ngrams.extend(extract_ngrams(words, n))
    return all_ngrams



# FREQUENCY MAPS
def build_ngram_frequency_map(lines, n=5):
    counter = Counter()
    for line in lines:
        normalized = normalize_text(line)
        words = extract_words(normalized, normalize=False)
        ngrams = extract_ngrams(words, n)
        counter.update(ngrams)
    return counter


# BUILD ALL N-GRAM FREQUENCIES IN SINGLE PASS
def build_all_ngram_frequency_maps(lines, sizes=[1, 2, 3, 4, 5]):
    counters = {n: Counter() for n in sizes}
    for line in lines:
        normalized = normalize_text(line)
        words = extract_words(normalized, normalize=False)
        for n in sizes:
            ngrams = extract_ngrams(words, n)
            counters[n].update(ngrams)
    return counters



# TOKEN EFFICIENCY STATS
def compute_efficiency_stats(lines):
    total_tokens = 0
    total_words = 0
    unique_tokens = set()
    for line in lines:
        tokens = tokenize_text(line)
        words = extract_words(line, normalize=True)
        total_tokens += len(tokens)
        total_words += len(words)
        unique_tokens.update(tokens)

    return {

        "total_tokens":
            total_tokens,

        "total_words":
            total_words,

        "tokens_per_word":
            round(total_tokens / max(total_words, 1), 3),

        "unique_token_ratio":
            round(len(unique_tokens) / max(total_tokens, 1), 3)
    }



# REDUNDANCY MINING
def compute_redundancy_scores(normal_freq, redundant_freq, min_redundant_frequency=2, threshold=1.5):
    scores = {}
    for phrase, redundant_count in redundant_freq.items():
        if redundant_count < min_redundant_frequency:
            continue
        normal_count = normal_freq.get(phrase, 0)
        # statistical score -> weight strategy varies by ngram length
        n = len(phrase.split())
        freq_ratio = math.log(redundant_count + 1) / math.log(normal_count + 2)
        if n <= 2:
            # short ngrams: frequency contrast dominates
            score = freq_ratio * math.log(redundant_count + 2) * (1 + 0.2 * n)
        elif n == 3:
            # mid-length: balanced blend of frequency ratio and length
            score = freq_ratio * math.sqrt(redundant_count + 1) * n
        else:
            # long ngrams (4-5): length is the dominant factor
            score = freq_ratio * n ** 1.8 * math.log(redundant_count + 1.5)
        if score >= threshold:
            scores[phrase] = {
                "type": "redundant_phrase",
                "source": "statistical",
                "redundancy_score": round(score, 3),
                "normal_frequency": normal_count,
                "redundant_frequency": redundant_count,
                "token_cost": token_count(phrase),
                "word_count": len(extract_words(phrase, normalize=False)),
                "ngram_size": len(phrase.split())
            }
    return scores



# KNOWLEDGE BASE
def build_knowledge_base(normal_lines, redundant_lines):
    knowledge = {}
    ngram_sizes = [1, 2, 3, 4, 5]
    
    print("Building n-gram frequencies for all sizes...")
    # Build all n-gram frequencies in single pass through each dataset
    normal_freqs = build_all_ngram_frequency_maps(normal_lines, ngram_sizes)
    redundant_freqs = build_all_ngram_frequency_maps(redundant_lines, ngram_sizes)
    
    # Compute redundancy scores for all n-gram sizes
    for n in ngram_sizes:
        print(f"Computing redundancy scores for {n}-grams...")
        scores = compute_redundancy_scores(normal_freqs[n], redundant_freqs[n])
        for phrase, data in scores.items():
            if phrase not in knowledge or data["redundancy_score"] > knowledge[phrase]["redundancy_score"]:
                knowledge[phrase] = data
    
    return knowledge



# SAVE KNOWLEDGE BASE
def save_knowledge_base(data,path):

    with open(path, "w", encoding="utf-8") as file:

        json.dump(data, file, indent=4, ensure_ascii=False)


# MAIN
def main():

    print("Loading datasets...")

    normal_lines = load_dataset(
        "data/normal_prompts.txt"
    )

    redundant_lines = load_dataset(
        "data/redundant_prompts.txt"
    )

    print("Building knowledge base...")

    knowledge = build_knowledge_base(

        normal_lines,
        redundant_lines
    )

    print("Saving knowledge base...")

    save_knowledge_base(

        knowledge,

        "knowledge/redundant_patterns.json"
    )

    print("\nKnowledge base generated.")

    print(
        f"Patterns discovered: "
        f"{len(knowledge)}"
    )

    print("\nTop patterns:\n")

    sorted_patterns = sorted(

        knowledge.items(),

        key=lambda x:
            x[1]["redundancy_score"],

        reverse=True
    )

    for phrase, data in sorted_patterns[:20]:

        print(f"Phrase: {phrase}")

        print(
            f"Score: "
            f"{data['redundancy_score']}"
        )

        print(
            f"Token cost: "
            f"{data['token_cost']}"
        )

        print(
            f"Redundant freq: "
            f"{data['redundant_frequency']}"
        )

        print("-" * 40)

    print("\nEfficiency Stats:\n")

    print(

        "NORMAL:",

        compute_efficiency_stats(
            normal_lines
        )
    )

    print(

        "REDUNDANT:",

        compute_efficiency_stats(
            redundant_lines
        )
    )


if __name__ == "__main__":
    main()