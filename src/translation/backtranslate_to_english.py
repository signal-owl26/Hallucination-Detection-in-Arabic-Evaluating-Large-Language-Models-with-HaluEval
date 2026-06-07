# Back-translate Arabic Dataset to English

## 1. Setup

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


## 2. Configuration

BACKTRANSLATION_MODEL = "gpt-4o-mini"

PROJECT_ROOT = Path("../..").resolve()

INPUT_FILE = PROJECT_ROOT / "data" / "arabic" / "intermediate"/"qa_data_arabic2.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "arabic" /"intermediate"/ "qa_data_backtranslated2.json"

SYSTEM_PROMPT_FILE = PROJECT_ROOT / "prompts" / "translation_en_system_prompt.txt"
USER_PROMPT_FILE = PROJECT_ROOT / "prompts" / "translation_en_user_prompt.txt"



load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Back-translation model:", BACKTRANSLATION_MODEL)
print("Input file:", INPUT_FILE)
print("Output file:", OUTPUT_FILE)


## 3. Load Data and Prompts

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
    system_prompt = f.read()

with open(USER_PROMPT_FILE, "r", encoding="utf-8") as f:
    user_prompt_template = f.read()

print(f"Loaded {len(data)} Arabic samples.")


## 4. Back-translation Function

def backtranslate_sample(entry):
    """
    Back-translate one Arabic dataset sample into English while preserving the JSON structure.
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
        model=BACKTRANSLATION_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    backtranslated_text = response.choices[0].message.content.strip()
    backtranslated_sample = json.loads(backtranslated_text)

    return backtranslated_sample


def write_jsonl(data, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

## 5. Run Back-translation

backtranslated_data = []
failed_backtranslations = 0

# For the full run, use data.
for entry in tqdm(data, desc="Back-translating samples"):
    try:
        backtranslated_entry = backtranslate_sample(entry)
        backtranslated_data.append(backtranslated_entry)

    except Exception as error:
        failed_backtranslations += 1
        print(f"Error back-translating sample id={entry.get('id')}: {error}")
        time.sleep(2)


## 6. Save Back-translated Dataset

write_jsonl(backtranslated_data, OUTPUT_FILE)

print(f"Input Arabic samples: {len(data)}")
print(f"Back-translated samples: {len(backtranslated_data)}")
print(f"Failed back-translations: {failed_backtranslations}")
print("Saved back-translated dataset to:")
print(OUTPUT_FILE)
