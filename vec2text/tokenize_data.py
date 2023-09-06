import random
from typing import Callable, Dict

import datasets
import torch

from vec2text.models import InversionModel


def tokenize_function(
    tokenizer,
    embedder_tokenizer,
    text_column_name,
    max_seq_length,
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
    assert "embedder_input_ids" in batch.keys(), f"invalid keys {batch.keys()}"
    assert "embedder_attention_mask" in batch.keys(), f"invalid keys {batch.keys()}"
    assert hasattr(model, "call_embedding_model")

    model_device = next(model.parameters()).device
    with torch.no_grad():
        batch["frozen_embeddings"] = model.call_embedding_model(
            input_ids=torch.tensor(batch["embedder_input_ids"], device=model_device),
            attention_mask=torch.tensor(
                batch["embedder_attention_mask"], device=model_device
            ),
        )

    return batch


def whiten_embeddings_torch(
    embeddings: torch.Tensor, n_sample=2 * 10**4
) -> torch.Tensor:
    # https://github.com/Jun-jie-Huang/WhiteningBERT/blob/d1ef06ae00ac5c3d869a028dd817a68833870d72/sentence_transformers/pooling_utils.py#L40-L47
    print("\t[whitening] mean")
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    embeddings_sample = embeddings[:n_sample]
    print("\t[whitening]  cov")
    cov = torch.mm((embeddings_sample - mu).t(), embeddings_sample - mu)
    print("\t[whitening] svd")
    u, s, vt = torch.svd(cov)
    print("\t[whitening] W")
    W = torch.mm(u, torch.diag(1 / torch.sqrt(s)))
    print("\t[whitening] output")
    embeddings = torch.mm(embeddings - mu, W)
    return embeddings


def whiten_embedded_dataset(dataset_dict: datasets.DatasetDict) -> datasets.DatasetDict:
    for key in dataset_dict:
        dataset = dataset_dict[key]
        print(f"whitening split – {key} (len {len(dataset)})")
        embeddings = torch.tensor(dataset["frozen_embeddings"])
        whitened_embeddings = whiten_embeddings_torch(embeddings=embeddings)
        whitened_embeddings = whitened_embeddings.cpu().tolist()
        dataset_dict[key] = dataset_dict[key].add_column(
            "frozen_embeddings_whitened", whitened_embeddings
        )
    return dataset_dict


def randomly_truncate_inputs(
    inputs: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    # randomly truncates inputs. assumes last input is a pad token.
    seq_length = inputs["input_ids"].shape[1]
    new_length = random.randint(1, seq_length - 1)
    pos = random.randint(0, seq_length - new_length)
    truncated_inputs = {k: v[:, pos : pos + new_length] for k, v in inputs.items()}
    truncated_inputs_with_pad = {
        k: torch.cat((truncated_inputs[k], inputs[k][:, -1, None]), dim=1)
        for k, v in inputs.items()
    }
    # TODO fix eos and bos?
    return truncated_inputs_with_pad
