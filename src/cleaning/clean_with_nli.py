# NLI-based Dataset Cleaning

## 1. Setup

import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

## 2. Configuration

PROJECT_ROOT = Path("../..").resolve()

INPUT_FILE = PROJECT_ROOT / "data" / "original" / "halu_eval_qa.json"

REMOVED_FILE = PROJECT_ROOT / "data" / "cleaned" / "removed_not_hallucinations.json"
CLEANED_FILE = PROJECT_ROOT / "data" / "cleaned" / "qa_data_cleaned.json"
ALL_RESULTS_FILE = PROJECT_ROOT / "data" / "cleaned" / "classification_results.json"

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

## 3. Load Models

tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(DEVICE)
nli_model.eval()

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

print("NLI model loaded:", NLI_MODEL_NAME)
print("Embedding model loaded:", EMBEDDING_MODEL_NAME)

## 4. Helper Functions

def get_nli_label(premise: str, hypothesis: str) -> str:
    """
    Predict the NLI relation between a premise and a hypothesis.

    """

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.no_grad():
        logits = nli_model(**inputs).logits

    predicted_id = int(torch.argmax(logits, dim=1).item())
    label = nli_model.config.id2label[predicted_id].lower()

    return label

# Extracts all the numbers form a text.
def extract_numbers(text):
    return re.findall(r"\d+", text) 

# If the hallucinated answer has different numbers than the one in the right answer then it is a valid hallucination 
def has_numeric_conflict(right_answer, hallucinated_answer):
    right_numbers = extract_numbers(right_answer)
    hall_numbers = extract_numbers(hallucinated_answer)

    if right_numbers and hall_numbers and right_numbers != hall_numbers:
        return True
    return False


# Check if an important entity from the right answer is missing from the hallucinated answer, if so then it is a hallucination
def has_missing_entity(right_answer, hallucinated_answer):
    right_tokens = right_answer.lower().split()
    hallu = hallucinated_answer.lower()


    for tokens in right_tokens:
        if len(tokens) <= 2:
            continue
        if tokens not in hallu: 
            return True
    return False

# Computes the cosine similarity
def compute_cosine_similarity(input1, input2):
    emb1 = embedder.encode(input1, convert_to_tensor=True)
    emb2 = embedder.encode(input2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)) # Compute the cosine similarity between the two embeddings

## 5. Hallucination Detection Function

def classify_sample(question, knowledge, right_answer, hallucinated_answer):

    # if numeric conflict, then it is hallucination
    if has_numeric_conflict(right_answer, hallucinated_answer):
        return "HALLUCINATION", {
            "reason": "numeric mismatch",
            "gold": right_answer,
            "hallucinated": hallucinated_answer
        }



    # if it is missing entities then it is hallucination 
    if has_missing_entity(right_answer, hallucinated_answer):
        return "HALLUCINATION", {
            "reason": "missing key entity",
            "right": right_answer,
            "hallucinated": hallucinated_answer
        }

    # NLI Check
    premise = f"Question: {question}\nContext: {knowledge}\nGold answer: {right_answer}"
    hypothesis = hallucinated_answer
    nli_label = get_nli_label(premise, hypothesis)
    similarity = compute_cosine_similarity(right_answer, hallucinated_answer)

    
    # If Contradiction -> hallucination
    if nli_label == "contradiction":
        return "HALLUCINATION", {
            "nli": nli_label,
            "similarity": similarity,
            "reason": "contradiction"
        }

    # if Neutral -> ambiguous -> hallucination
    if nli_label == "neutral":
        
        return "HALLUCINATION", {
            "nli": nli_label,
            "similarity": similarity,
            "reason": "neutral unsupported"
        }

    # otherwise if Entailment -> not hallucination
    return "NOT_HALLUCINATION", {
        "nli": nli_label,
        "similarity": similarity,
        "reason": "entailment"
    }

## 6. Run Cleaning

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

# Make sure every original sample has an ID.
# If not, IDs are created from the position in the file.
for index, entry in enumerate(data):
    if "id" not in entry:
        entry["id"] = index

print(f"Loaded {len(data)} samples from {INPUT_FILE}")

classification_results = []

for entry in tqdm(data, desc="Classifying samples"):
    label, debug = classify_sample(
        question=entry["question"],
        knowledge=entry["knowledge"],
        right_answer=entry["right_answer"],
        hallucinated_answer=entry["hallucinated_answer"],
    )

    classification_results.append({
        "id": entry["id"],
        "label": label,
        "debug": debug,
        "knowledge": entry["knowledge"],
        "question": entry["question"],
        "right_answer": entry["right_answer"],
        "hallucinated_answer": entry["hallucinated_answer"],
    })

## 7. Save Classification Results

ALL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(ALL_RESULTS_FILE, "w", encoding="utf-8") as f:
    for sample in classification_results:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Saved {len(classification_results)} classification results to:")
print(ALL_RESULTS_FILE)

## 7. Save Removed Samples

removed_samples = [
    entry for entry in classification_results
    if entry["label"] == "NOT_HALLUCINATION"
]

REMOVED_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(REMOVED_FILE, "w", encoding="utf-8") as f:
    for sample in removed_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Saved {len(removed_samples)} removed samples to:")
print(REMOVED_FILE)

## 8. Save Cleaned Dataset

ids_to_remove = {entry["id"] for entry in removed_samples}

cleaned_data = []

for entry in data:
    if entry["id"] in ids_to_remove:
        continue

    cleaned_entry = {
        "id": len(cleaned_data),
        "knowledge": entry["knowledge"],
        "question": entry["question"],
        "right_answer": entry["right_answer"],
        "hallucinated_answer": entry["hallucinated_answer"],
    }

    cleaned_data.append(cleaned_entry)

CLEANED_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(CLEANED_FILE, "w", encoding="utf-8") as f:
    for sample in cleaned_data:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Original samples: {len(data)}")
print(f"Removed samples: {len(removed_samples)}")
print(f"Cleaned samples: {len(cleaned_data)}")
print("Saved cleaned dataset to:")
print(CLEANED_FILE)