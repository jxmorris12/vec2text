import pytest

import torch
import vec2text

@pytest.fixture
def corrector_ada(scope="session"):
    return vec2text.load_corrector("text-embedding-ada-002")

def test_invert_embeddings(corrector_ada):
    embeddings = torch.randn(
        (3, corrector_ada.model.embedder_dim),
        device=vec2text.models.model_utils.device,
        dtype=torch.float32
    )
    inverted_texts = vec2text.invert_embeddings(
        embeddings=embeddings,
        corrector=corrector_ada
    )
    assert len(inverted_texts) == len(embeddings)


def test_invert_strings(corrector_ada):
    test_strings = [
        "Mage (foaled April 18, 2020)[2] is an American Thoroughbred racehorse who won the 2023 Kentucky Derby",
        "“The arts are not a way to make a living. They are a very human way of making life more bearable. Practicing an art, no matter how well or badly, is a way to make your soul grow, for heaven's sake.",
    ]
    inverted_texts = vec2text.invert_strings(
        test_strings,
        corrector=corrector_ada,
    )
    assert inverted_texts == [
        'Mystic Rider (born March 23, 1922) is an American thoroughbred horse and a winner of the Kentucky Derby in 2018.[1]', 
        "Art and living are not a way to make a living, they are a way to make a living. ''The way they make a living is to make a living, for the human soul, to appreciate the pleasures of life, if you will."
    ]

def test_invert_strings_beam(corrector_ada):
    test_strings = [
        "Mage (foaled April 18, 2020)[2] is an American Thoroughbred racehorse who won the 2023 Kentucky Derby",
        "“The arts are not a way to make a living. They are a very human way of making life more bearable. Practicing an art, no matter how well or badly, is a way to make your soul grow, for heaven's sake.",
    ]
    inverted_texts = vec2text.invert_strings(
        test_strings,
        corrector=corrector_ada,
        num_recursive_steps=10,
        sequence_beam_width=8,
    )
    assert inverted_texts == [
        'Magician (foaled April 23, 2018)[2] is an American Thoroughbred racehorse who won the Kentucky Derby in 2023', 
        "''Arts are not a way to make a living, they are a way to make the human soul grow. Practicing an art, no matter how badly, is a way to make your life in heaven a little more bearable. And very much so."
    ]