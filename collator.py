import collections
from typing import Any, Dict, List, Optional, Union

import torch
import transformers

# def pad_tensor_to_length(the_list: List[int], length: int = 0, value: int = 0) -> List[int]:
#     num_pads = length - len(the_list)
#     if num_pads == 0:
#         return torch.tensor(the_list, dtype=torch.long)
#     else:
#         return torch.tensor(the_list + [value] * num_pads, dtype=torch.long)


# def cut_padding(batch: Dict[str, torch.Tensor], pad_token: int) -> Dict[str, torch.Tensor]:
#     """Truncates doc if some stuff is all padding at the end."""
#     assert 'input_ids' in batch
#     assert 'attention_mask' in batch
#     #
#     b, s = batch['input_ids'].shape
#     #
#     all_padding = (batch['input_ids'] == pad_token).all(dim=0)
#     if all_padding.sum() == 0:
#         return batch
#     #
#     padding_start = all_padding.int().argmax()
#     batch['input_ids'] = batch['input_ids'][:, :padding_start]
#     batch['attention_mask'] = batch['attention_mask'][:, :padding_start]
#     return batch


class CustomCollator(transformers.DataCollatorWithPadding):
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        stacked_features = collections.defaultdict(list)
        #
        #
        #
        for ex in features:
            for k, v in ex.items():
                stacked_features[k].append(v)
        #
        #
        # stack other features
        ex = {}
        for k, v in stacked_features.items():
            # TODO why are these not tensors already since we tokenized with
            #   return_tensors='pt'?
            ex[k] = torch.tensor(v)

        # TODO: call cut_padding and test if it speeds up the code
        return ex
