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

def test_inversion_model_dpr(fake_data):
    embedder, embedder_tokenizer = (
        load_embedder_and_tokenizer(name="dpr")
    )
    model = InversionModel(
        embedder=embedder,
        encoder_decoder=load_encoder_decoder(
            model_name="t5-small",
        ),
        num_repeat_tokens=6,
    )

    # test model forward.
    model(inputs=fake_data)

    # test generate.
    #  (todo)
