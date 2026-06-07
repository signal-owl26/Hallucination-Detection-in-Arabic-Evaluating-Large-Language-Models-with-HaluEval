# Translate cleaned English dataset to Arabic

## 1. Setup

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


## 2. Configuration

TRANSLATION_MODEL = "gpt-4o-mini"

PROJECT_ROOT = Path("../..").resolve()

INPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "qa_data_cleaned2.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "arabic" / "intermediate"/"qa_data_arabic2.json"

SYSTEM_PROMPT_FILE = PROJECT_ROOT / "prompts" / "translation_ar_system_prompt.txt"
USER_PROMPT_FILE = PROJECT_ROOT / "prompts" / "translation_ar_user_prompt.txt"



FIELDS_TO_CHECK = [
    "knowledge",
    "question",
    "right_answer",
    "hallucinated_answer",
]

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Translation model:", TRANSLATION_MODEL)


## 3. Load Data and Prompts

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
    system_prompt = f.read()

with open(USER_PROMPT_FILE, "r", encoding="utf-8") as f:
    user_prompt_template = f.read()

print(f"Loaded {len(data)} samples from {INPUT_FILE}")

## 4. Arabic-Script Helper Functions

def is_arabic_character(character):
    """
    Check whether a character belongs to an Arabic Unicode block.
    """

    codepoint = ord(character)

    arabic_ranges = [
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ]

    for start, end in arabic_ranges:
        if start <= codepoint <= end:
            return True

    return False


def has_non_arabic_letters(text):
    """
    Return True if the text contains alphabetic letters that are not Arabic.

    Numbers, punctuation, and spaces are allowed.
    """

    for character in str(text):
        if character.isalpha() and not is_arabic_character(character):
            return True

    return False


def sample_has_non_arabic_letters(sample):
    """
    Check whether any translated field still contains non-Arabic letters.
    """

    for field in FIELDS_TO_CHECK:
        if has_non_arabic_letters(sample.get(field, "")):
            return True

    return False

## 5. Translation Function

def translate_sample(entry):
    """
    Translate one dataset sample from English to Arabic while preserving the JSON structure.
    """

    sample = {
        "id": entry["id"],
        "knowledge": entry["knowledge"],
        "question": entry["question"],
        "right_answer": entry["right_answer"],
        "hallucinated_answer": entry["hallucinated_answer"],
    }

    sample_json = json.dumps(sample, ensure_ascii=False, indent=2)

    user_prompt = f"{user_prompt_template}\n\n{sample_json}"

    response = client.chat.completions.create(
        model=TRANSLATION_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    translated_text = response.choices[0].message.content.strip()
    translated_sample = json.loads(translated_text)

    return translated_sample


def write_jsonl(data, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

## 6. Run Translation and Arabic-Script Filtering

translated_data = []
removed_by_script_filter = 0
failed_translations = 0

# For the full run, use data.
for entry in tqdm(data, desc="Translating samples"):
    try:
        translated_entry = translate_sample(entry)

        if sample_has_non_arabic_letters(translated_entry):
            removed_by_script_filter += 1
            continue

        translated_data.append(translated_entry)

    except Exception as error:
        failed_translations += 1
        print(f"Error translating sample id={entry.get('id')}: {error}")
        time.sleep(2)


## 7. Save Arabic Dataset

write_jsonl(translated_data, OUTPUT_FILE)

print(f"Input samples: {len(data)}")
print(f"Kept Arabic samples: {len(translated_data)}")
print(f"Removed by Arabic-script filtering: {removed_by_script_filter}")
print(f"Failed translations: {failed_translations}")
print("Saved Arabic dataset to:")
print(OUTPUT_FILE)