from pathlib import Path
from core.optimizer import (
    load_knowledge_base,
    load_prompt_from_file,
    save_prompt_to_file,
    interactive_optimize_prompt,
    show_final_report
)


BASE_DIR = Path(__file__).resolve().parent

PROMPT_FILE = BASE_DIR / "data" / "prompt_to_optimize.txt"
KNOWLEDGE_FILE = BASE_DIR / "knowledge" / "redundant_patterns.json"


def main():
    print("Loading prompt...")
    original_prompt = load_prompt_from_file(PROMPT_FILE)

    print("Loading knowledge base...")
    knowledge_base = load_knowledge_base(KNOWLEDGE_FILE)

    optimized_prompt, accepted_changes = interactive_optimize_prompt(
        original_prompt,
        knowledge_base
    )

    show_final_report(
        original_prompt,
        optimized_prompt,
        accepted_changes
    )

    choice = input("\nSave optimized prompt to file? Type y/n: ").strip().lower()

    if choice == "y":
        save_prompt_to_file(PROMPT_FILE, optimized_prompt)
        print("Prompt saved successfully.")
    else:
        print("Prompt was not saved.")


if __name__ == "__main__":
    main()