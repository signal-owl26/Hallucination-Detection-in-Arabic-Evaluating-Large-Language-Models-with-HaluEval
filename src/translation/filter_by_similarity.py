# Filter Translated Dataset by Semantic Similarity

## 1. Setup

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


## 2. Configuration

PROJECT_ROOT = Path("../..").resolve()

ORIGINAL_ENGLISH_FILE = PROJECT_ROOT / "data" / "cleaned" / "qa_data_cleaned.json"
ARABIC_FILE = PROJECT_ROOT / "data" / "arabic" / "intermediate"/"qa_data_arabic.json"
BACKTRANSLATED_FILE = PROJECT_ROOT / "data" / "arabic" / "intermediate"/ "qa_data_backtranslated.json"

SIMILARITY_RESULTS_FILE = PROJECT_ROOT / "data" / "arabic" /"intermediate"/ "semantic_similarity_results.json"
SIMILARITY_RESULTS_085_FILE = PROJECT_ROOT / "data" / "arabic" / "intermediate"/"semantic_similarity_results_0_85.json"

FINAL_ARABIC_FILE = PROJECT_ROOT / "data" / "arabic" / "final" /"final_arabic_dataset.json"
FINAL_ENGLISH_FILE = PROJECT_ROOT / "data" / "arabic" / "final"/"final_english_dataset.json"

SIMILARITY_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.85

FIELDS_TO_COMPARE = [
    "knowledge",
    "question",
    "right_answer",
    "hallucinated_answer",
]

print("Similarity model:", SIMILARITY_MODEL_NAME)
print("Similarity threshold:", SIMILARITY_THRESHOLD)


## 3. Load Model

embedder = SentenceTransformer(SIMILARITY_MODEL_NAME)


## 4. Helper Functions

def compute_cosine_similarity(text1, text2):
    """
    Compute cosine similarity between two text fields.
    """

    embedding1 = embedder.encode(str(text1), convert_to_tensor=True)
    embedding2 = embedder.encode(str(text2), convert_to_tensor=True)

    return float(util.cos_sim(embedding1, embedding2).item())


def passes_threshold(result):
    """
    Return True if all cosine scores are greater than or equal to the threshold.
    """

    return (
        result["knowledge_cosine"] >= SIMILARITY_THRESHOLD
        and result["question_cosine"] >= SIMILARITY_THRESHOLD
        and result["right_answer_cosine"] >= SIMILARITY_THRESHOLD
        and result["hallucinated_answer_cosine"] >= SIMILARITY_THRESHOLD
    )

def read_jsonl(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(data, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

## 5. Load Data

english_data = read_jsonl(ORIGINAL_ENGLISH_FILE)
arabic_data = read_jsonl(ARABIC_FILE)
backtranslated_data = read_jsonl(BACKTRANSLATED_FILE)

print(f"Loaded English samples: {len(english_data)}")
print(f"Loaded Arabic samples: {len(arabic_data)}")
print(f"Loaded back-translated samples: {len(backtranslated_data)}")


## 6. Index Data by ID

english_by_id = {
    str(entry["id"]): entry
    for entry in english_data
}

arabic_by_id = {
    str(entry["id"]): entry
    for entry in arabic_data
}

backtranslated_by_id = {
    str(entry["id"]): entry
    for entry in backtranslated_data
}


## 7. Compute Similarity Scores

similarity_results = []
similarity_results_085 = []

for sample_id, backtranslated_entry in tqdm(
    backtranslated_by_id.items(),
    desc="Computing similarity"
):
    if sample_id not in english_by_id:
        continue

    if sample_id not in arabic_by_id:
        continue

    english_entry = english_by_id[sample_id]

    result = {
        "id": int(sample_id),
        "knowledge_cosine": compute_cosine_similarity(
            english_entry["knowledge"],
            backtranslated_entry["knowledge"],
        ),
        "question_cosine": compute_cosine_similarity(
            english_entry["question"],
            backtranslated_entry["question"],
        ),
        "right_answer_cosine": compute_cosine_similarity(
            english_entry["right_answer"],
            backtranslated_entry["right_answer"],
        ),
        "hallucinated_answer_cosine": compute_cosine_similarity(
            english_entry["hallucinated_answer"],
            backtranslated_entry["hallucinated_answer"],
        ),
    }

    similarity_results.append(result)

    if passes_threshold(result):
        similarity_results_085.append(result)


## 8. Create Final Arabic and English Datasets


keep_ids = {
    str(result["id"])
    for result in similarity_results_085
}

final_arabic_data = [
    entry.copy()
    for entry in arabic_data
    if str(entry["id"]) in keep_ids
]

final_english_data = [
    entry.copy()
    for entry in english_data
    if str(entry["id"]) in keep_ids
]

for new_id, entry in enumerate(final_arabic_data):
    entry["id"] = new_id

for new_id, entry in enumerate(final_english_data):
    entry["id"] = new_id


## 9. Save Results

write_jsonl(similarity_results, SIMILARITY_RESULTS_FILE)
write_jsonl(similarity_results_085, SIMILARITY_RESULTS_085_FILE)
write_jsonl(final_arabic_data, FINAL_ARABIC_FILE)
write_jsonl(final_english_data, FINAL_ENGLISH_FILE)

print(f"Samples with all scores >= {SIMILARITY_THRESHOLD}: {len(similarity_results_085)}")
print(f"Final Arabic samples: {len(final_arabic_data)}")
print(f"Final English samples: {len(final_english_data)}")

print("Saved all similarity scores to:")
print(SIMILARITY_RESULTS_FILE)

print("Saved passed similarity scores to:")
print(SIMILARITY_RESULTS_085_FILE)

print("Saved final Arabic dataset to:")
print(FINAL_ARABIC_FILE)

print("Saved final English dataset to:")
print(FINAL_ENGLISH_FILE)
