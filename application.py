# Enhanced GUI with 2 modes: Threat Detection and Prompt Optimization

import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
import sys

import tiktoken

# Add core to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.detect import detect
from core.optimizer import (
    load_knowledge_base,
    generate_suggestions,
    apply_suggestion,
    normalize_spaces,
    count_tokens,
)

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge" / "redundant_patterns.json"
THREAT_KB_FILE = BASE_DIR / "knowledge" / "threat_patterns.json"

encoding = tiktoken.get_encoding("cl100k_base")


def clean_word(word):
    return re.sub(r"[^\w]", "", word).lower()


def load_kb(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


class EnhancedMP1App:
    def __init__(self, root):
        self.root = root
        self.root.title("Prompt Analyzer - Threat Detection & Optimization")
        self.root.geometry("1200x800")

        self.knowledge_base = load_kb(KNOWLEDGE_FILE)
        self.threat_kb = load_kb(THREAT_KB_FILE)
        
        self.current_optimization_step = 0
        self.current_suggestions = []
        self.ignored_targets = set()
        self.accepted_changes = []
        self.current_optimized_prompt = ""

        self.create_widgets()

    def create_widgets(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="Prompt Analyzer - Threat Detection & Optimization",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tab 1: Input
        self.input_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.input_frame, text="Input Prompt")
        self.create_input_tab()

        # Tab 2: Threat Detection
        self.threat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.threat_frame, text="Detect Threats")
        self.create_threat_tab()

        # Tab 3: Optimization
        self.optimize_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.optimize_frame, text="Optimize Prompt")
        self.create_optimize_tab()

    def create_input_tab(self):
        input_label = tk.Label(
            self.input_frame,
            text="Enter your prompt here:",
            font=("Arial", 11, "bold")
        )
        input_label.pack(anchor="w", padx=15, pady=10)

        self.prompt_text = tk.Text(
            self.input_frame,
            height=20,
            wrap="word",
            font=("Arial", 11)
        )
        self.prompt_text.pack(fill="both", expand=True, padx=15, pady=5)

        # Sample text
        sample = "Could you please please summarize this article in a very very short and concise way for me please?"
        self.prompt_text.insert("1.0", sample)

        button_frame = tk.Frame(self.input_frame)
        button_frame.pack(pady=10)

        clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_input,
            width=20,
            bg="#f4cccc"
        )
        clear_btn.pack(side="left", padx=5)

    def create_threat_tab(self):
        display_label = tk.Label(
            self.threat_frame,
            text="Threat Analysis Report:",
            font=("Arial", 11, "bold")
        )
        display_label.pack(anchor="w", padx=15, pady=10)

        self.threat_display = tk.Text(
            self.threat_frame,
            height=15,
            wrap="word",
            font=("Consolas", 10)
        )
        self.threat_display.pack(fill="both", expand=True, padx=15, pady=5)

        # Tags for threats
        self.threat_display.tag_configure("critical", foreground="red", font=("Consolas", 10, "bold"))
        self.threat_display.tag_configure("high", foreground="orange", font=("Consolas", 10, "bold"))
        self.threat_display.tag_configure("medium", foreground="gold", font=("Consolas", 10))
        self.threat_display.tag_configure("low", foreground="blue", font=("Consolas", 10))
        self.threat_display.tag_configure("label", font=("Consolas", 10, "bold"))

        button_frame = tk.Frame(self.threat_frame)
        button_frame.pack(pady=10)

        analyze_btn = tk.Button(
            button_frame,
            text="Analyze for Threats",
            command=self.analyze_threats,
            width=30,
            bg="#d9ead3",
            font=("Arial", 10, "bold")
        )
        analyze_btn.pack(side="left", padx=5)

    def create_optimize_tab(self):
        display_label = tk.Label(
            self.optimize_frame,
            text="Optimization Suggestions & Results:",
            font=("Arial", 11, "bold")
        )
        display_label.pack(anchor="w", padx=15, pady=10)

        # Suggestion display
        self.optimize_display = tk.Text(
            self.optimize_frame,
            height=12,
            wrap="word",
            font=("Consolas", 10)
        )
        self.optimize_display.pack(fill="both", expand=True, padx=15, pady=5)

        # Tags for optimization
        self.optimize_display.tag_configure("redundant", foreground="red", font=("Consolas", 10, "bold"))
        self.optimize_display.tag_configure("accepted", foreground="green", font=("Consolas", 10, "bold"))
        self.optimize_display.tag_configure("info", font=("Consolas", 10))

        button_frame = tk.Frame(self.optimize_frame)
        button_frame.pack(pady=10)

        optimize_btn = tk.Button(
            button_frame,
            text="Start Optimization",
            command=self.start_optimization,
            width=20,
            bg="#d9ead3",
            font=("Arial", 10, "bold")
        )
        optimize_btn.pack(side="left", padx=5)

        accept_btn = tk.Button(
            button_frame,
            text="Accept (y)",
            command=lambda: self.handle_optimization_response("y"),
            width=15,
            bg="#c6e9a8",
            font=("Arial", 10)
        )
        accept_btn.pack(side="left", padx=5)

        reject_btn = tk.Button(
            button_frame,
            text="Reject (n)",
            command=lambda: self.handle_optimization_response("n"),
            width=15,
            bg="#f8cbad",
            font=("Arial", 10)
        )
        reject_btn.pack(side="left", padx=5)

        stop_btn = tk.Button(
            button_frame,
            text="Stop",
            command=self.stop_optimization,
            width=15,
            bg="#f4cccc",
            font=("Arial", 10)
        )
        stop_btn.pack(side="left", padx=5)

    def get_input_text(self):
        return self.prompt_text.get("1.0", "end-1c").strip()

    def clear_input(self):
        self.prompt_text.delete("1.0", tk.END)

    def analyze_threats(self):
        prompt = self.get_input_text()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt first.")
            return

        self.threat_display.delete("1.0", tk.END)

        try:
            # Run threat detection
            result = detect(prompt, kb_path=str(self.threat_kb), verbose=False)

            # Display results
            danger_score = result.get("danger_score", 0)
            verdict = result.get("verdict", "UNKNOWN")
            threats = result.get("threats", [])

            # Header
            header = f"{danger_score:.0f}% danger - Verdict: {verdict}\n"
            header += "=" * 60 + "\n\n"
            self.threat_display.insert(tk.END, header, "label")

            # Threat type summary
            if threats:
                threat_types = set()
                for t in threats:
                    threat_types.add(t["type"])

                summary = ""
                for threat_type in sorted(threat_types):
                    if threat_type == "pattern":
                        summary += "• Dangerous pattern detected\n"
                    elif threat_type == "statistical":
                        summary += "• Dangerous learned phrase detected\n"
                    elif threat_type == "unicode":
                        summary += "• Dangerous unicode detected\n"

                self.threat_display.insert(tk.END, summary + "\n", "info")

                # Detailed threats
                self.threat_display.insert(tk.END, f"Detected Threats ({len(threats)}):\n", "label")
                self.threat_display.insert(tk.END, "-" * 60 + "\n", "info")

                for t in threats:
                    severity = t.get("severity", "low").lower()
                    label = t["label"]
                    evidence = t["evidence"]

                    threat_line = f"[{severity.upper():8}] {label}\n"
                    self.threat_display.insert(tk.END, threat_line, severity)
                    
                    evidence_line = f"             Evidence: \"{evidence}\"\n"
                    self.threat_display.insert(tk.END, evidence_line, "info")
            else:
                self.threat_display.insert(tk.END, "No threats detected.\n", "info")

        except Exception as e:
            messagebox.showerror("Error", f"Error during threat analysis:\n{str(e)}")

    def start_optimization(self):
        prompt = self.get_input_text()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt first.")
            return

        self.optimize_display.delete("1.0", tk.END)
        self.current_optimized_prompt = normalize_spaces(prompt)
        self.current_suggestions = []
        self.ignored_targets = set()
        self.accepted_changes = []
        self.current_optimization_step = 0

        self.show_next_optimization_step()

    def show_next_optimization_step(self):
        self.optimize_display.delete("1.0", tk.END)

        # Generate suggestions
        self.current_suggestions = generate_suggestions(
            self.current_optimized_prompt,
            self.knowledge_base,
            self.ignored_targets
        )

        if not self.current_suggestions:
            # Show final report
            self.show_optimization_report()
            return

        suggestion = self.current_suggestions[0]
        self.current_optimization_step += 1

        # Display suggestion
        header = f"STEP {self.current_optimization_step}\n"
        header += "=" * 60 + "\n\n"
        self.optimize_display.insert(tk.END, header, "label")

        info = f"Current prompt:\n{self.current_optimized_prompt}\n\n"
        self.optimize_display.insert(tk.END, info, "info")

        if suggestion["type"] == "direct_repetition":
            sugg_text = f"Type: Direct repetition\n"
            sugg_text += f"Replace: '{suggestion['target']}' -> '{suggestion['replacement']}'\n"
        else:
            sugg_text = f"Type: Knowledge base word/phrase\n"
            sugg_text += f"Remove: '{suggestion['target']}'\n"
            sugg_text += f"Score: {suggestion['score']}\n"

        sugg_text += f"N-gram size: {suggestion['ngram_size']}\n"
        sugg_text += f"Estimated saved tokens: {suggestion['saved_tokens']}\n"
        sugg_text += f"Reason: {suggestion['reason']}\n\n"

        self.optimize_display.insert(tk.END, sugg_text, "redundant")

        preview_text = f"Preview after change:\n{suggestion['preview']}\n"
        self.optimize_display.insert(tk.END, preview_text, "info")

    def handle_optimization_response(self, response):
        if not self.current_suggestions:
            messagebox.showinfo("Info", "No active optimization session.")
            return

        suggestion = self.current_suggestions[0]

        if response == "y":
            self.current_optimized_prompt = apply_suggestion(
                self.current_optimized_prompt,
                suggestion
            )
            self.current_optimized_prompt = normalize_spaces(self.current_optimized_prompt)
            self.accepted_changes.append(suggestion)
        elif response == "n":
            self.ignored_targets.add(suggestion["target"])

        self.show_next_optimization_step()

    def stop_optimization(self):
        self.show_optimization_report()

    def show_optimization_report(self):
        self.optimize_display.delete("1.0", tk.END)

        original_prompt = self.get_input_text()
        original_tokens = count_tokens(original_prompt)
        optimized_tokens = count_tokens(self.current_optimized_prompt)
        saved_tokens = original_tokens - optimized_tokens

        header = "OPTIMIZATION COMPLETE\n"
        header += "=" * 60 + "\n\n"
        self.optimize_display.insert(tk.END, header, "label")

        result_text = f"Optimized prompt:\n{self.current_optimized_prompt}\n\n"
        self.optimize_display.insert(tk.END, result_text, "info")

        stats = f"TOKEN STATISTICS\n"
        stats += f"Original tokens:  {original_tokens}\n"
        stats += f"Optimized tokens: {optimized_tokens}\n"
        stats += f"Saved tokens:     {saved_tokens}\n"

        if original_tokens > 0:
            reduction = (saved_tokens / original_tokens) * 100
            stats += f"Reduction:        {reduction:.2f}%\n"

        self.optimize_display.insert(tk.END, stats, "label")

        if self.accepted_changes:
            changes_text = f"\nACCEPTED CHANGES ({len(self.accepted_changes)}):\n"
            self.optimize_display.insert(tk.END, changes_text, "accepted")

            for change in self.accepted_changes:
                if change["type"] == "direct_repetition":
                    change_line = f"- Replace '{change['target']}' -> '{change['replacement']}'\n"
                else:
                    change_line = f"- Remove '{change['target']}' (saved {change['saved_tokens']} tokens)\n"
                self.optimize_display.insert(tk.END, change_line, "accepted")
        else:
            no_changes = "\nNo changes were accepted.\n"
            self.optimize_display.insert(tk.END, no_changes, "info")

    def clear_all(self):
        self.prompt_text.delete("1.0", tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedMP1App(root)
    root.mainloop()