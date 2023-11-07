from typing import Callable, Dict

import torch
import transformers

from vec2text.models import InversionModel


def tokenize_function(
    tokenizer: transformers.PreTrainedTokenizer,
    embedder_tokenizer: transformers.PreTrainedTokenizer,
    text_column_name: str,
    max_seq_length: int,
    padding: bool = False,
) -> Callable[[Dict], Dict]:
    def tokenize_function_inner(examples) -> Dict[str, torch.Tensor]:
        output = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )

        # copy to 'labels' for language modeling loss
        # but set padding to -100
        # github.com/huggingface/transformers/blob/cbe63949d76efd153a1f389f38fe9ce1287e06b0/src/transformers/models/t5/modeling_t5.py#L1504-L1507
        output["labels"] = [
            [
                (-100 if token_id == tokenizer.pad_token_id else token_id)
                for token_id in ids
            ]
            for ids in output["input_ids"]
        ]
        embedder_output = embedder_tokenizer(
            examples[text_column_name],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        embedder_output = {f"embedder_{k}": v for k, v in embedder_output.items()}

        output["length"] = [
            (torch.tensor(input_ids) != tokenizer.pad_token_id).sum().item()
            for input_ids in output["input_ids"]
        ]

        return {**output, **embedder_output}

    return tokenize_function_inner


def tokenize_function_llama_chat(
    tokenizer,
    embedder_tokenizer,
    text_column_name,
    max_seq_length,
    padding: bool = False,
) -> Callable[[Dict], Dict]:
    """Use special tokenization for LLAMA chat models."""

    def tokenize_function_inner(examples) -> Dict[str, torch.Tensor]:
        if "prefix" not in examples:
            # hacky way to turn datasets into the right format for LLAMA chat.
            # "real" prompt datasets like one_million_paired_instructions
            # have "prefix" and "suffix" already.
            #
            # so this is only for evaluation datasets that may not have
            # actual prefix-suffix pairing.
            #
            examples["prefix"] = [""] * len(examples[text_column_name])
            examples["suffix"] = examples[text_column_name]

        output = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )

        # copy to 'labels' for language modeling loss
        # but set padding to -100
        # github.com/huggingface/transformers/blob/cbe63949d76efd153a1f389f38fe9ce1287e06b0/src/transformers/models/t5/modeling_t5.py#L1504-L1507
        output["labels"] = [
            [
                (-100 if token_id == tokenizer.pad_token_id else token_id)
                for token_id in ids
            ]
            for ids in output["input_ids"]
        ]
        embedder_output = embedder_tokenizer(
            text=[
                f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n {instruction} [/INST]"
                for (system_message, instruction) in zip(
                    examples["prefix"], examples["suffix"]
                )
            ],
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        embedder_output = {f"embedder_{k}": v for k, v in embedder_output.items()}

        output["length"] = [
            (torch.tensor(input_ids) != tokenizer.pad_token_id).sum().item()
            for input_ids in output["input_ids"]
        ]

        return {**output, **embedder_output}

    return tokenize_function_inner


def embed_dataset_batch(model: InversionModel, batch: Dict) -> Dict:
    assert "input_ids" in batch.keys(), f"invalid keys {batch.keys()}"
    assert hasattr(model, "call_embedding_model")

    input_ids = batch["input_ids"]
    inputs_str = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    emb_input_ids = model.embedder_tokenizer(
        inputs_str,
        max_length=model.config.max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        batch["frozen_embeddings"] = model.call_embedding_model(**emb_input_ids)
    return batch
