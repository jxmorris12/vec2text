from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import transformers


@dataclass
class DataCollatorForCorrection:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels, and hypotheses.

    Based off of hf DataCollatorForSeq2Seq:
        github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L517
    """

    tokenizer: transformers.PreTrainedTokenizer
    label_pad_token_id: int = -100
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_label_length = max(len(l) for l in labels)
        if self.pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        padding_side = self.tokenizer.padding_side

        if "hypothesis_input_ids" in features[0].keys():
            max_hypothesis_length = max(
                map(lambda d: len(d["hypothesis_input_ids"]), features)
            )
        else:
            max_hypothesis_length = 0
        hypothesis_features = []
        regular_features = []
        for feature in features:
            ### pad labels
            remainder = [self.label_pad_token_id] * (
                max_label_length - len(feature["labels"])
            )
            if isinstance(feature["labels"], list):
                feature["labels"] = (
                    feature["labels"] + remainder
                    if padding_side == "right"
                    else remainder + feature["labels"]
                )
            elif padding_side == "right":
                feature["labels"] = np.concatenate(
                    [feature["labels"], remainder]
                ).astype(np.int64)
            else:
                feature["labels"] = np.concatenate(
                    [remainder, feature["labels"]]
                ).astype(np.int64)
            #### add to lists
            regular_features.append(
                {k: v for k, v in feature.items() if not k.startswith("hypothesis_")}
            )

            hypothesis_features.append(
                {
                    k.replace("hypothesis_", ""): v
                    for k, v in feature.items()
                    if k.startswith("hypothesis_")
                }
            )

        new_features = self.tokenizer.pad(
            regular_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if max_hypothesis_length > 0:
            hypothesis_features = self.tokenizer.pad(
                hypothesis_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
            hypothesis_features = {
                f"hypothesis_{k}": v for k, v in hypothesis_features.items()
            }
            return {**new_features, **hypothesis_features}
        else:
            return new_features
