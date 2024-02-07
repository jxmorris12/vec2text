import os
from typing import Any, Dict

import torch
import torch.nn as nn
import transformers
from sentence_transformers import SentenceTransformer

EMBEDDER_MODEL_NAMES = [
    "bert",
    "bert__random_init",
    "contriever",
    "dpr",
    "gtr_base",
    "gtr_base__random_init",
    "medicalai/ClinicalBERT",
    "gtr_large",
    "ance_tele",
    "dpr_st",
    "gtr_base_st",
    "paraphrase-distilroberta",
    "sentence-transformers/all-MiniLM-L6-v2",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "nomic-ai/nomic-embed-text-v1",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
]


FREEZE_STRATEGIES = ["decoder", "encoder_and_decoder", "encoder", "none"]
EMBEDDING_TRANSFORM_STRATEGIES = ["repeat"]


def get_device():
    """
    Function that checks
    for GPU availability and returns
    the appropriate device.
    :return: torch.device
    """
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device


device = get_device()


def disable_dropout(model: nn.Module):
    dropout_modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in dropout_modules:
        m.p = 0.0
    print(
        f"Disabled {len(dropout_modules)} dropout modules from model type {type(model)}"
    )


def freeze_params(model: nn.Module):
    total_num_params = 0
    for name, params in model.named_parameters():
        params.requires_grad = False
        total_num_params += params.numel()
    # print(f"Froze {total_num_params} params from model type {type(model)}")


def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def max_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.max(dim=1).values
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def stack_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.reshape((B, S * D))  # stack along seq length
    assert pooled_outputs.shape == (B, S * D)
    return pooled_outputs


def load_embedder_and_tokenizer(name: str, torch_dtype: str, **kwargs):
    # TODO make abstract/argparse for it etc.
    # name = "gpt2" #### <--- TEMP. For debugging. Delete!
    model_kwargs = {
        "low_cpu_mem_usage": True,  # Not compatible with DeepSpeed
        "output_hidden_states": False,
    }

    if name == "dpr":
        # model = SentenceTransformer("sentence-transformers/facebook-dpr-question_encoder-multiset-base")
        model = transformers.DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
    elif name == "dpr_st":
        # TODO figure out why model w/ sentence transformers gives different results.
        model = SentenceTransformer(
            "sentence-transformers/facebook-dpr-question_encoder-multiset-base"
        )
        tokenizer = model.tokenizer
    elif name == "contriever":
        model = transformers.AutoModel.from_pretrained(
            "facebook/contriever", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/contriever")
    elif name == "bert":
        model = transformers.AutoModel.from_pretrained(
            "bert-base-uncased", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    elif name == "bert__random_init":
        config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
        model = transformers.AutoModel.from_config(config)
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    elif name == "gtr_base":
        model = transformers.AutoModel.from_pretrained(
            "sentence-transformers/gtr-t5-base", **model_kwargs
        ).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
    elif name == "gtr_large":
        model = transformers.AutoModel.from_pretrained(
            "sentence-transformers/gtr-t5-large", **model_kwargs
        ).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-large"
        )
    elif name == "gtr_base__random_init":
        config = transformers.AutoConfig.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
        model = transformers.AutoModel.from_config(config).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
    elif name == "gtr_base_st":
        model = SentenceTransformer("sentence-transformers/gtr-t5-base")
        tokenizer = model.tokenizer
    elif name == "gtr_large":
        model = SentenceTransformer("sentence-transformers/gtr-t5-large")
        tokenizer = model.tokenizer
    elif name == "ance_tele":
        model = transformers.AutoModel.from_pretrained(
            "OpenMatch/ance-tele_nq_psg-encoder", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "OpenMatch/ance-tele_nq_psg-encoder"
        )
    elif name == "paraphrase-distilroberta":
        model = transformers.AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1"
        )
    elif name == "medicalai/ClinicalBERT":
        model = transformers.AutoModel.from_pretrained(
            "medicalai/ClinicalBERT", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    elif name.startswith("gpt2"):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name,
            **model_kwargs,
        )
        # model.to_bettertransformer()
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
    elif name.startswith("meta-llama/Llama-2-70b"):
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_config = transformers.AutoConfig.from_pretrained(
            name,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        model.eval()
    elif name.startswith("meta-llama/"):
        if torch_dtype == "float32":
            torch_dtype = torch.float32
        elif torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name,
            **model_kwargs,
            token=os.environ.get("LLAMA_TOKEN"),
            torch_dtype=torch_dtype,
            **kwargs,
        )
        # if torch_dtype is not torch.float32:
        #     model.to_bettertransformer()
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
    elif name.startswith("sentence-transformers/"):
        model = SentenceTransformer(name)
        tokenizer = model.tokenizer
    elif name.startswith("nomic-ai/nomic-embed-text-v1"):
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        )
        tokenizer = model.tokenizer
    else:
        print(f"WARNING: Trying to initialize from unknown embedder {name}")
        model = transformers.AutoModel.from_pretrained(name, **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    # model = torch.compile(model)
    return model, tokenizer


def load_encoder_decoder(
    model_name: str, lora: bool = False
) -> transformers.AutoModelForSeq2SeqLM:
    model_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }
    if lora:
        model_kwargs.update(
            {
                "load_in_8bit": True,
                "device_map": "auto",
            }
        )
    return transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name, **model_kwargs
    )


def load_tokenizer(name: str, max_length: int) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name,
        padding="max_length",
        truncation="max_length",
        max_length=max_length,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable super annoying warning:
    # https://github.com/huggingface/transformers/issues/22638
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tokenizer
