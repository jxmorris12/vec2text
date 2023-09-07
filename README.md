# vec2text

This library contains code for doing text embedding inversion. We can train various architectures that reconstruct text sequences from embeddings as well as run pre-trained models. This repository contains code for the paper "Text Embeddings Reveal (Almost)
As Much As Text".

To get started, install this on PyPI:

```bash
pip install vec2text
```

[Link to Colab Demo](https://colab.research.google.com/drive/14RQFRF2It2Kb8gG3_YDhP_6qE0780L8h?usp=sharing)

## Usage

The library can be used to embed text and then invert it, or invert directly from embeddings. First you'll need to construct a `Corrector` object which wraps the necessary models, embedders, and tokenizers:

### Load a model via `load_corrector`

```python
corrector = vec2text.load_corrector("text-embedding-ada-002")
```

### Invert text with `invert_strings`

```python
vec2text.invert_strings(
    [
        "Jack Morris is a PhD student at Cornell Tech in New York City",
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity"
    ],
    corrector=corrector,
)
['Morris is a PhD student at Cornell University in New York City',
 'It was the age of incredulity, the age of wisdom, the age of apocalypse, the age of apocalypse, it was the age of faith, the age of best faith, it was the age of foolishness']
```

By default, this will make a single guess (using the hypothesizer). For better results, you can make multiple steps:

```python
vec2text.invert_strings(
    [
        "Jack Morris is a PhD student at Cornell Tech in New York City",
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity"
    ],
    corrector=corrector,
    num_recursive_steps=20,
)
['Jack Morris is a PhD student in tech at Cornell University in New York City',
 'It was the best time of the epoch, it was the worst time of the epoch, it was the best time of the age of wisdom, it was the age of incredulity, it was the age of betrayal']
```

And for even better results, you can increase the size of the search space by setting `sequence_beam_width` to a positive integer:

```python
vec2text.invert_strings(
    [
        "Jack Morris is a PhD student at Cornell Tech in New York City",
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity"
    ],
    corrector=corrector,
    num_recursive_steps=20,
    sequence_beam_width=4,
)
['Jack Morris is a PhD student at Cornell Tech in New York City',
 'It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity']
```

Note that this technique has to store `sequence_beam_width * sequence_beam_width` hypotheses at each step, so if you set it too high, you'll run out of GPU memory.

### Invert embeddings with `invert_embeddings`

If you only have embeddings, you can invert them directly:

```python
import torch

def get_embeddings_openai(text_list, model="text-embedding-ada-002") -> torch.Tensor:
    response = openai.Embedding.create(
        input=text_list,
        model=model,
        encoding_format="float",  # override default base64 encoding...
    )
    outputs.extend([e["embedding"] for e in response["data"]])
    return torch.tensor(outputs)


embeddings = get_embeddings_openai([
       "Jack Morris is a PhD student at Cornell Tech in New York City",
       "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity"
])


vec2text.invert_embeddings(
    embeddings=embeddings.cuda(),
    corrector=corrector
)
['Morris is a PhD student at Cornell University in New York City',
 'It was the age of incredulity, the age of wisdom, the age of apocalypse, the age of apocalypse, it was the age of faith, the age of best faith, it was the age of foolishness']
```

This function also takes the same optional hyperparameters, `num_steps` and `sequence_beam_width`.

## Pre-trained models

Currently we only support models for inverting OpenAI `text-embedding-ada-002` embeddings but are hoping to add more soon. (We can provide the GTR inverters used in the paper upon request.)

Our models come in one of two forms: a zero-step 'hypothesizer' model that makes a guess for what text is from an embedding and a 'corrector' model that iteratively corrects and re-embeds text to bring it closer to the target embedding. We also support *sequence-level beam search* which makes multiple corrective guesses at each step and takes the one closest to the ground-truth embedding.



### pre-commit

```pip install isort black flake8 mypy --upgrade```

```pre-commit run --all```



### Cite our paper

Please cite our paper! We will add a citation once paper is officially online :)