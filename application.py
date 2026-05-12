# TODO: implement GUI logic using Tkinter

import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import tiktoken


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge" / "redundant_patterns.json"

encoding = tiktoken.get_encoding("cl100k_base")


def load_knowledge_base(path):
    #Loads redundant patterns from JSON file
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        messagebox.showerror(
            "Error",
            f"Knowledge base file not found:\n{path}"
        )
        return {}


def clean_word(word):
    return re.sub(r"[^\w]", "", word).lower()


def count_tokens(text):
    return len(encoding.encode(text))


def get_tokens(text):
    #Returns token ids and decoded token text
    token_ids = encoding.encode(text)

    decoded_tokens = []
    for token_id in token_ids:
        decoded_tokens.append(encoding.decode([token_id]))

    return token_ids, decoded_tokens


def get_redundant_words(knowledge_base, min_score=4.0):
    #Extracts redundant words
    redundant_words = {}

    for pattern, data in knowledge_base.items():
        if data.get("type") == "redundant_word":
            score = data.get("score", 0)

            if score >= min_score:
                redundant_words[pattern] = score

    return redundant_words


def get_repeated_phrases(knowledge_base):
    #Extracts repeated phrases
    repeated_phrases = {}

    for pattern, data in knowledge_base.items():
        if data.get("type") == "repeated_phrase":
            repeated_phrases[pattern] = data.get("frequency", 0)

    return repeated_phrases


class MP1App:
    def __init__(self, root):
        self.root = root
        self.root.title("Tokenization Visualizer")
        self.root.geometry("1000x700")

        self.knowledge_base = load_knowledge_base(KNOWLEDGE_FILE)

        self.redundant_words = get_redundant_words(
            self.knowledge_base,
            min_score=4.0
        )

        self.repeated_phrases = get_repeated_phrases(self.knowledge_base)

        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(
            self.root,
            text="Tokenization and Redundancy Analyzer",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        description_label = tk.Label(
            self.root,
            text="Write a prompt, analyze tokenization and see redundant words highlighted in red.",
            font=("Arial", 10)
        )
        description_label.pack(pady=5)

        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="both", expand=False, padx=15, pady=10)

        input_label = tk.Label(
            input_frame,
            text="Input Prompt:",
            font=("Arial", 11, "bold")
        )
        input_label.pack(anchor="w")

        self.prompt_text = tk.Text(
            input_frame,
            height=8,
            wrap="word",
            font=("Arial", 11)
        )
        self.prompt_text.pack(fill="both", expand=True)

        self.prompt_text.tag_configure(
            "redundant",
            foreground="red",
            font=("Arial", 11, "bold")
        )

        self.prompt_text.tag_configure(
            "repeated",
            foreground="white",
            background="red",
            font=("Arial", 11, "bold")
        )

        sample_prompt = (
            "I honestly really really need you to very carefully explain this extremely complicated code in a very very simple way."
        )

        self.prompt_text.insert("1.0", sample_prompt)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        analyze_button = tk.Button(
            button_frame,
            text="Analyze Prompt",
            command=self.analyze_prompt,
            width=20,
            bg="#d9ead3"
        )
        analyze_button.grid(row=0, column=0, padx=5)

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_all,
            width=20,
            bg="#f4cccc"
        )
        clear_button.grid(row=0, column=1, padx=5)

        results_frame = tk.Frame(self.root)
        results_frame.pack(fill="both", expand=True, padx=15, pady=10)

        left_frame = tk.Frame(results_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)

        right_frame = tk.Frame(results_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)

        stats_label = tk.Label(
            left_frame,
            text="Token Statistics:",
            font=("Arial", 11, "bold")
        )
        stats_label.pack(anchor="w")

        self.stats_text = tk.Text(
            left_frame,
            height=10,
            wrap="word",
            font=("Consolas", 10)
        )
        self.stats_text.pack(fill="both", expand=True)

        tokens_label = tk.Label(
            right_frame,
            text="Tokenization Output:",
            font=("Arial", 11, "bold")
        )
        tokens_label.pack(anchor="w")

        self.tokens_text = tk.Text(
            right_frame,
            height=10,
            wrap="word",
            font=("Consolas", 10)
        )
        self.tokens_text.pack(fill="both", expand=True)

        threats_label = tk.Label(
            self.root,
            text="Redundancy Report:",
            font=("Arial", 11, "bold")
        )
        threats_label.pack(anchor="w", padx=15)

        self.report_text = tk.Text(
            self.root,
            height=7,
            wrap="word",
            font=("Consolas", 10)
        )
        self.report_text.pack(fill="both", expand=True, padx=15, pady=5)

    def analyze_prompt(self):
        prompt = self.prompt_text.get("1.0", "end-1c")

        if not prompt.strip():
            messagebox.showwarning("Warning", "Please enter a prompt first.")
            return

        self.clear_previous_results()

        token_ids, decoded_tokens = get_tokens(prompt)

        word_count = len(re.findall(r"\b\w+\b", prompt))
        token_count_value = len(token_ids)

        tokens_per_word = token_count_value / max(word_count, 1)

        redundant_found = self.highlight_redundant_words(prompt)
        repeated_found = self.highlight_repeated_phrases(prompt)

        self.show_statistics(
            token_count_value,
            word_count,
            tokens_per_word,
            redundant_found,
            repeated_found
        )

        self.show_tokens(token_ids, decoded_tokens)

        self.show_report(redundant_found, repeated_found)

    def clear_previous_results(self):
        self.stats_text.delete("1.0", tk.END)
        self.tokens_text.delete("1.0", tk.END)
        self.report_text.delete("1.0", tk.END)

        self.prompt_text.tag_remove("redundant", "1.0", tk.END)
        self.prompt_text.tag_remove("repeated", "1.0", tk.END)

    def highlight_redundant_words(self, prompt):
        #Highlights redundant words in red
        
        redundant_found = []

        words = re.finditer(r"\b\w+\b", prompt)

        for match in words:
            word = match.group()
            cleaned = clean_word(word)

            if cleaned in self.redundant_words:
                start_index = f"1.0 + {match.start()} chars"
                end_index = f"1.0 + {match.end()} chars"

                self.prompt_text.tag_add(
                    "redundant",
                    start_index,
                    end_index
                )

                redundant_found.append({
                    "word": word,
                    "score": self.redundant_words[cleaned]
                })

        return redundant_found

    def highlight_repeated_phrases(self, prompt):
        #Highlights repeated phrases with red

        repeated_found = []

        for phrase, frequency in self.repeated_phrases.items():
            pattern = r"\b" + re.escape(phrase) + r"\b"

            for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
                start_index = f"1.0 + {match.start()} chars"
                end_index = f"1.0 + {match.end()} chars"

                self.prompt_text.tag_add(
                    "repeated",
                    start_index,
                    end_index
                )

                repeated_found.append({
                    "phrase": match.group(),
                    "frequency": frequency
                })

        return repeated_found

    def show_statistics(
        self,
        token_count_value,
        word_count,
        tokens_per_word,
        redundant_found,
        repeated_found
    ):
        stats = ""
        stats += f"Token count:       {token_count_value}\n"
        stats += f"Word count:        {word_count}\n"
        stats += f"Tokens per word:   {tokens_per_word:.2f}\n"
        stats += f"Redundant words:   {len(redundant_found)}\n"
        stats += f"Repeated phrases:  {len(repeated_found)}\n"

        if token_count_value <= 20:
            efficiency = "Good"
        elif token_count_value <= 50:
            efficiency = "Medium"
        else:
            efficiency = "High token usage"

        stats += f"Efficiency level:  {efficiency}\n"

        self.stats_text.insert(tk.END, stats)

    def show_tokens(self, token_ids, decoded_tokens):
        #Displays token id and decoded token text
        
        for index, (token_id, token_text) in enumerate(
            zip(token_ids, decoded_tokens),
            start=1
        ):
            safe_token_text = token_text.replace("\n", "\\n")
            self.tokens_text.insert(
                tk.END,
                f"{index}. ID={token_id} | text='{safe_token_text}'\n"
            )

    def show_report(self, redundant_found, repeated_found):
        if not redundant_found and not repeated_found:
            self.report_text.insert(
                tk.END,
                "No redundant words or repeated phrases detected.\n"
            )
            return

        if repeated_found:
            self.report_text.insert(tk.END, "Repeated phrases detected:\n")

            for item in repeated_found:
                self.report_text.insert(
                    tk.END,
                    f"- '{item['phrase']}' | frequency in dataset: {item['frequency']}\n"
                )

            self.report_text.insert(tk.END, "\n")

        if redundant_found:
            self.report_text.insert(tk.END, "Redundant words detected:\n")

            for item in redundant_found:
                self.report_text.insert(
                    tk.END,
                    f"- '{item['word']}' | redundancy score: {item['score']}\n"
                )

    def clear_all(self):
        self.prompt_text.delete("1.0", tk.END)
        self.clear_previous_results()


if __name__ == "__main__":
    root = tk.Tk()
    app = MP1App(root)
    root.mainloop()