import os
import torch
import transformers
from dotenv import load_dotenv

from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv(override=True)

MODEL_MAP = {
    # OpenAI API
    "gpt4o": {
        "provider": "openai",
        "model_name": "gpt-4o",
    },

    # DeepSeek API
    "deepseek-chat": {
        "provider": "deepseek",
        "model_name": "deepseek-chat",
    },

    "deepseek-reasoner": {
        "provider": "deepseek",
        "model_name": "deepseek-reasoner",
    },

    # Hugging Face local models
    "mistral": {
        "provider": "huggingface",
        "model_name": "mistralai/Ministral-8B-Instruct-2410",
    },
    "ace": {
        "provider": "huggingface",
        "model_name": "FreedomIntelligence/AceGPT-v2-32B-Chat",
    },
    "llama": {
        "provider": "llama_pipeline",
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },

}


def setup_model(model_key):
    """
    Load the selected model or API client.
    """
    if model_key not in MODEL_MAP:
        raise ValueError(
            f"Unknown model_key: {model_key}. "
            f"Available models: {list(MODEL_MAP.keys())}"
        )

    model_info = MODEL_MAP[model_key]
    provider = model_info["provider"]
    model_name = model_info["model_name"]

    api_client = None
    hf_tokenizer = None
    hf_model = None
    llama_pipeline = None

    if provider == "openai":
        api_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )

    elif provider == "deepseek":
        api_client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    elif provider == "huggingface":
        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if hf_tokenizer.pad_token_id is None:
            hf_tokenizer.pad_token_id = hf_tokenizer.eos_token_id

    elif provider == "llama_pipeline":
        llama_pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    print("\nModel setup:")
    print("model_key:", model_key)
    print("provider:", provider)
    print("model_name:", model_name)
    print("api_client set:", api_client is not None)
    print("hf_tokenizer set:", hf_tokenizer is not None)
    print("hf_model set:", hf_model is not None)
    print("llama_pipeline set:", llama_pipeline is not None)

    return {
        "model_key": model_key,
        "provider": provider,
        "model_name": model_name,
        "api_client": api_client,
        "hf_tokenizer": hf_tokenizer,
        "hf_model": hf_model,
        "llama_pipeline": llama_pipeline,
    }