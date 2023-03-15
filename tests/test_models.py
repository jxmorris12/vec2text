from typing import Dict

import torch
import transformers
import pytest

from models import (
    load_encoder_decoder, load_embedder_and_tokenizer, InversionModel, FREEZE_STRATEGIES, MODEL_NAMES
)


@pytest.fixture
def fake_data() -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {
        'embedder_input_ids': input_ids,
        'embedder_attention_mask': attention_mask,
        # 
        'labels': input_ids,
    }

def __test_embedding_model(
        fake_data: Dict[str, torch.Tensor],
        embedder_model_name: str,
        no_grad: bool,
        freeze_strategy: str,
        embedder_fake_with_zeros: bool,
        use_frozen_embeddings_as_input: bool,
    ):
    encoder_decoder_model_name = "t5-small"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        encoder_decoder_model_name,
        padding=True,
        truncation='max_length',
        max_length=16,
    )
    embedder, embedder_tokenizer = (
        load_embedder_and_tokenizer(name=embedder_model_name)
    )
    model = InversionModel(
        embedder=embedder,
        embedder_tokenizer=embedder_tokenizer,
        tokenizer=tokenizer,
        encoder_decoder=load_encoder_decoder(
            model_name=encoder_decoder_model_name,
        ),
        num_repeat_tokens=6,
        embedder_no_grad=no_grad,
        freeze_strategy=freeze_strategy,
        embedder_fake_with_zeros=embedder_fake_with_zeros,
        use_frozen_embeddings_as_input=use_frozen_embeddings_as_input,
    )

    # test model forward.
    model(**fake_data)

    # test generate.
    generation_kwargs = {
        'max_length': 4,
        'num_beams': 1,
        'do_sample': False,
    }
    model.generate(
        inputs=fake_data, generation_kwargs=generation_kwargs
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_inversion_models(fake_data, model_name):
    __test_embedding_model(fake_data, model_name, True, "none", False, False)
    __test_embedding_model(fake_data, model_name, False, "none", False, False)


@pytest.mark.parametrize("freeze_strategy", FREEZE_STRATEGIES)
def test_inversion_model_frozen(fake_data, freeze_strategy):
    __test_embedding_model(fake_data, "dpr", True, freeze_strategy, False, False)


def test_inversion_model_zeros(fake_data):
    __test_embedding_model(fake_data, "dpr", True, "none", True, False)


def test_inversion_model_frozen_embeddings_input(fake_data):
    with pytest.raises(AssertionError):
        __test_embedding_model(fake_data, "gtr_base", True, "none", False, True)
    
    fake_data["frozen_embeddings"] = torch.randn((1, 768), dtype=torch.float32)
    __test_embedding_model(fake_data, "gtr_base", True, "none", False, True)