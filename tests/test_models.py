from typing import Dict

import torch
import transformers
import pytest

from models import load_encoder_decoder, load_embedder_and_tokenizer, InversionModel


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

def __test_embedding_model(fake_data, embedding_model_name, no_grad):
    embedder, embedder_tokenizer = (
        load_embedder_and_tokenizer(name=embedding_model_name)
    )
    model = InversionModel(
        embedder=embedder,
        encoder_decoder=load_encoder_decoder(
            model_name="t5-small",
        ),
        num_repeat_tokens=6,
        embedder_no_grad=no_grad,
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


@pytest.mark.parametrize("model_name", ["dpr", "ance_tele", "gtr_base"])
def test_inversion_model_gtr(fake_data, model_name):
    __test_embedding_model(fake_data, model_name, True)
    __test_embedding_model(fake_data, model_name, False)