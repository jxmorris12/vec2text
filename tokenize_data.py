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

        # copy to 'labels' for language modeling loss
        # but set padding to -100
        # github.com/huggingface/transformers/blob/cbe63949d76efd153a1f389f38fe9ce1287e06b0/src/transformers/models/t5/modeling_t5.py#L1504-L1507
        output['labels'] = torch.where(
            output['input_ids'] == tokenizer.pad_token_id,
            -100,
            output['input_ids']
        )

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