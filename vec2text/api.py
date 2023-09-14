import copy
from typing import List

import torch
import transformers

import vec2text
from vec2text.models.model_utils import device

SUPPORTED_MODELS = ["text-embedding-ada-002"]


def load_corrector(embedder: str) -> vec2text.trainers.Corrector:
    """Gets the Corrector object for the given embedder.

    For now, we just support inverting OpenAI Ada 002 embeddings; we plan to
    expand this support over time.
    """
    assert (
        embedder in SUPPORTED_MODELS
    ), f"embedder to invert `{embedder} not in list of supported models: {SUPPORTED_MODELS}`"

    inversion_model = vec2text.models.InversionModel.from_pretrained(
        "jxm/vec2text__openai_ada002__msmarco__msl128__hypothesizer"
    )
    model = vec2text.models.CorrectorEncoderModel.from_pretrained(
        "jxm/vec2text__openai_ada002__msmarco__msl128__corrector"
    )

    inversion_trainer = vec2text.trainers.InversionTrainer(
        model=inversion_model,
        train_dataset=None,
        eval_dataset=None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            inversion_model.tokenizer,
            label_pad_token_id=-100,
        ),
    )

    # backwards compatibility stuff
    model.config.dispatch_batches = None
    corrector = vec2text.trainers.Corrector(
        model=model,
        inversion_trainer=inversion_trainer,
        args=None,
        data_collator=vec2text.collator.DataCollatorForCorrection(
            tokenizer=inversion_trainer.model.tokenizer
        ),
    )
    return corrector


def invert_embeddings(
    embeddings: torch.Tensor,
    corrector: vec2text.trainers.Corrector,
    num_steps: int = None,
    sequence_beam_width: int = 0,
) -> List[str]:
    corrector.inversion_trainer.model.eval()
    corrector.model.eval()

    gen_kwargs = copy.copy(corrector.gen_kwargs)
    gen_kwargs["min_length"] = 1
    gen_kwargs["max_length"] = 128

    if num_steps is None:
        assert (
            sequence_beam_width == 0
        ), "can't set a nonzero beam width without multiple steps"

        regenerated = corrector.inversion_trainer.generate(
            inputs={
                "frozen_embeddings": embeddings,
            },
            generation_kwargs=gen_kwargs,
        )
    else:
        corrector.return_best_hypothesis = sequence_beam_width > 0
        regenerated = corrector.generate(
            inputs={
                "frozen_embeddings": embeddings,
            },
            generation_kwargs=gen_kwargs,
            num_recursive_steps=num_steps,
            sequence_beam_width=sequence_beam_width,
        )

    output_strings = corrector.tokenizer.batch_decode(
        regenerated, skip_special_tokens=True
    )
    return output_strings


def invert_strings(
    strings: List[str],
    corrector: vec2text.trainers.Corrector,
    num_steps: int = None,
    sequence_beam_width: int = 0,
) -> List[str]:
    inputs = corrector.embedder_tokenizer(
        strings,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True,
    )
    inputs = inputs.to(device)
    with torch.no_grad():
        frozen_embeddings = corrector.inversion_trainer.call_embedding_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
    return invert_embeddings(
        embeddings=frozen_embeddings,
        corrector=corrector,
        num_steps=num_steps,
        sequence_beam_width=sequence_beam_width,
    )
