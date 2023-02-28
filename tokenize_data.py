from typing import Dict

import torch


def tokenize_function(tokenizer, embedder_tokenizer, text_column_name, max_seq_length):
    def tokenize_function_inner(examples) -> Dict[str, torch.Tensor]:
        output = tokenizer(
            examples[text_column_name],
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors='pt',
        )
        output['labels'] = output['input_ids'] # copy to 'labels' for language modeling loss

        embedder_output = embedder_tokenizer(
            examples[text_column_name],
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors='pt'
        )
        embedder_output = { f'embedder_{k}': v for k,v in embedder_output.items() }

        return {**output, **embedder_output}
    return tokenize_function_inner