import os
import re
import json
import time
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from models import setup_model, MODEL_MAP


# ============================================================
# Output normalization
# ============================================================

def normalize_judgement(text: str, language: str):
    t = (text or "").strip()

    if not t:
        return None

    if language == "Ar":
        m = re.search(r"(نعم|لا)", t)
        return m.group(1) if m else None

    else:
        m = re.search(r"\b(Yes|No)\b", t, flags=re.IGNORECASE)

        if not m:
            return None

        return "Yes" if m.group(1).lower() == "yes" else "No"


# ============================================================
# Prompt construction
# ============================================================

def build_messages(knowledge, question, answer, instruction, language):
    if language == "Ar":
        messages = [
            {
                "role": "system",
                "content": (
                    "أنت كاشف هلوسة. يجب أن تحدد ما إذا كانت الإجابة تحتوي على هلوسة اعتمادًا فقط على المعرفة المقدمة. "
                    "يجب أن تكون إجابتك كلمة واحدة فقط: نعم أو لا."
                ),
            },
            {
                "role": "user",
                "content": (
                    instruction
                    + "\n\n#المعرفة#: " + knowledge
                    + "\n\n#السؤال#: " + question
                    + "\n#الإجابة#: " + answer
                    + "\n#حكمك#: "
                ),
            },
        ]

    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a hallucination detector. You MUST determine if the provided answer contains hallucination "
                    "based only on the provided knowledge. Your answer MUST be exactly one word: Yes or No."
                ),
            },
            {
                "role": "user",
                "content": (
                    instruction
                    + "\n\n#Knowledge#: " + knowledge
                    + "\n\n#Question#: " + question
                    + "\n#Answer#: " + answer
                    + "\n#Your Judgement#: "
                ),
            },
        ]

    return messages


# ============================================================
# Model response functions
# ============================================================

def get_qa_response_api(api_client, model_name, knowledge, question, answer, instruction, language):
    messages = build_messages(
        knowledge=knowledge,
        question=question,
        answer=answer,
        instruction=instruction,
        language=language,
    )

    max_tokens= 5000 if model_name=="deepseek-reasoner" else 5
    while True:
        try:
            response = api_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )

            raw_output = response.choices[0].message.content.strip()
            judgement = normalize_judgement(raw_output, language)

            return judgement

        except Exception as e:
            print(f"Error: {e}")
            print("Retrying in 20 seconds...")
            time.sleep(20)


def get_qa_response_huggingface(tokenizer, model, knowledge, question, answer, instruction, language):
    messages = build_messages(
        knowledge=knowledge,
        question=question,
        answer=answer,
        instruction=instruction,
        language=language,
    )

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    judgement = normalize_judgement(decoded, language)
    return judgement


def get_qa_response_llama_pipeline(pipe, knowledge, question, answer, instruction, language):
    messages = build_messages(
        knowledge=knowledge,
        question=question,
        answer=answer,
        instruction=instruction,
        language=language,
    )

    output = pipe(
        messages,
        max_new_tokens=5,
        do_sample=False,
        return_full_text=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    generated_text = output[0]["generated_text"]

    judgement = normalize_judgement(generated_text, language)
    return judgement


def get_qa_response(model_bundle, knowledge, question, answer, instruction, language):
    provider = model_bundle["provider"]
    model_name = model_bundle["model_name"]

    api_client = model_bundle["api_client"]
    hf_tokenizer = model_bundle["hf_tokenizer"]
    hf_model = model_bundle["hf_model"]
    llama_pipeline = model_bundle["llama_pipeline"]

    if provider in ["openai", "deepseek", "together"]:
        return get_qa_response_api(
            api_client=api_client,
            model_name=model_name,
            knowledge=knowledge,
            question=question,
            answer=answer,
            instruction=instruction,
            language=language,
        )

    elif provider == "huggingface":
        return get_qa_response_huggingface(
            tokenizer=hf_tokenizer,
            model=hf_model,
            knowledge=knowledge,
            question=question,
            answer=answer,
            instruction=instruction,
            language=language,
        )

    elif provider == "llama_pipeline":
        return get_qa_response_llama_pipeline(
            pipe=llama_pipeline,
            knowledge=knowledge,
            question=question,
            answer=answer,
            instruction=instruction,
            language=language,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================
# JSONL helpers
# ============================================================

def load_jsonl(input_path):
    data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            data.append(json.loads(line))

    return data


def dump_jsonl(record, output_path, append=True):
    mode = "a" if append else "w"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode, encoding="utf-8") as f:
        json_record = json.dumps(record, ensure_ascii=False)
        f.write(json_record + "\n")


# ============================================================
# Automatic path setup
# ============================================================

def get_paths(content_language, prompt_language, model_key):
    project_root = Path("../..").resolve()

    data_dir = project_root / "data"
    final_data_dir = data_dir / "arabic" / "final"
    prompt_dir = project_root / "prompts"
    evaluation_dir = data_dir / "evaluation"

    if content_language == "Ar":
        input_file = final_data_dir / "final_arabic_dataset copy.json"
    else:
        input_file = final_data_dir / "final_english_dataset copy.json"

    if prompt_language == "Ar":
        instruction_file = prompt_dir / "ar_evaluation_instruction.txt"
    else:
        instruction_file = prompt_dir / "en_evaluation_instruction.txt"

    # Folder name means: Prompt language + Content language
    # Example: En+Ar = English prompt, Arabic content
    config_name = f"{prompt_language}+{content_language}"

    output_dir = evaluation_dir / config_name
    output_file = output_dir / f"{model_key}.json"

    return input_file, instruction_file, output_file, config_name


# ============================================================
# Main evaluation method
# ============================================================

def evaluation_qa_dataset(
    file,
    instruction,
    output_path,
    language,
    model_bundle,
):
    data = load_jsonl(file)

    correct = 0
    incorrect = 0

    for i in tqdm(range(len(data))):
        sample_id = data[i]["id"]
        knowledge = data[i]["knowledge"]
        question = data[i]["question"]
        hallucinated_answer = data[i]["hallucinated_answer"]
        right_answer = data[i]["right_answer"]

        if sample_id % 2 == 0:
            answer = hallucinated_answer
            ground_truth = "نعم" if language == "Ar" else "Yes"

        else:
            answer = right_answer
            ground_truth = "لا" if language == "Ar" else "No"

        judgement = get_qa_response(
            model_bundle=model_bundle,
            knowledge=knowledge,
            question=question,
            answer=answer,
            instruction=instruction,
            language=language,
        )

        if judgement is None:
            record = {
                "id": sample_id,
                "knowledge": knowledge,
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "judgement": "failed!",
            }

            dump_jsonl(record, output_path, append=True)
            incorrect += 1

            print(f"sample {i} failed......")
            continue

        record = {
            "id": sample_id,
            "knowledge": knowledge,
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "judgement": judgement,
        }

        if judgement == ground_truth:
            correct += 1
        else:
            incorrect += 1

        dump_jsonl(record, output_path, append=True)

        print(f"sample {i} success......")

    accuracy = correct / len(data)

    print("=" * 80)
    print("Evaluation finished")
    print("=" * 80)
    print(f"Correct samples:   {correct}")
    print(f"Incorrect samples: {incorrect}")
    print(f"Accuracy:          {accuracy:.4f}")
    print("=" * 80)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_MAP.keys()),
        help="Model key from MODEL_MAP in models.py.",
    )

    parser.add_argument(
        "--content-language",
        required=True,
        choices=["En", "Ar"],
        help="Language of the dataset content. Use En or Ar.",
    )

    parser.add_argument(
        "--prompt-language",
        required=True,
        choices=["En", "Ar"],
        help="Language of the prompt and expected model judgement. Use En or Ar.",
    )

    parser.add_argument(
        "--cuda-device",
        default=None,
        help="Optional CUDA device, for example 0 or 1.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    input_file, instruction_file, output_path, config_name = get_paths(
        content_language=args.content_language,
        prompt_language=args.prompt_language,
        model_key=args.model,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            f"Use --overwrite if you want to replace it."
        )

    if output_path.exists() and args.overwrite:
        output_path.unlink()

    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_file}")

    if not instruction_file.exists():
        raise FileNotFoundError(f"Instruction file not found: {instruction_file}")

    with open(instruction_file, "r", encoding="utf-8") as f:
        instruction = f.read()

    print("=" * 80)
    print("Evaluation setup")
    print("=" * 80)
    print(f"Model:             {args.model}")
    print(f"Configuration:     {config_name}")
    print(f"Content language:  {args.content_language}")
    print(f"Prompt language:   {args.prompt_language}")
    print(f"Input file:        {input_file}")
    print(f"Instruction file:  {instruction_file}")
    print(f"Output file:       {output_path}")
    print("=" * 80)

    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Available:", torch.cuda.is_available())
    print("Count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model_bundle = setup_model(args.model)

    evaluation_qa_dataset(
        file=input_file,
        instruction=instruction,
        output_path=output_path,
        language=args.prompt_language,
        model_bundle=model_bundle,
    )


if __name__ == "__main__":
    main()