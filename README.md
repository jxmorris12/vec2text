# vec2text

<img src="https://github.com/jxmorris12/vec2text-gif/blob/master/vec2text_v3.gif" width="500" />

This library contains code for doing text embedding inversion. We can train various architectures that reconstruct text sequences from embeddings as well as run pre-trained models. This repository contains code for the paper "Text Embeddings Reveal (Almost)
As Much As Text".

To get started, install this on PyPI:

```bash
pip install vec2text
```

[Link to Colab Demo](https://colab.research.google.com/drive/14RQFRF2It2Kb8gG3_YDhP_6qE0780L8h?usp=sharing)

### Development

If you're training a model you'll need to set up nltk:
```python
import nltk
nltk.download('punkt')
```

Before pushing any code, please run precommit:
```bash
pre-commit run --all
```


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
    num_steps=20,
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
    num_steps=20,
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

### Interpolation

You can mix two embeddings together for interesting results. Given embeddings of the previous two inputs, we can invert their mean:

```python
vec2text.invert_embeddings(
    embeddings=embeddings.mean(dim=0, keepdim=True).cuda(),
    corrector=corrector
)
['Morris was in the age of physics, the age of astronomy, the age of physics, the age of physics PhD at New York']
```

Or do linear interpolation (this isn't particularly interesting, feel free to submit a PR with a cooler example):

```python
import numpy as np

for alpha in np.arange(0.0, 1.0, 0.1):
  mixed_embedding = torch.lerp(input=embeddings[0], end=embeddings[1], weight=alpha)
  text = vec2text.invert_embeddings(
      embeddings=mixed_embedding[None].cuda(),
      corrector=corrector,
      num_steps=20,
      sequence_beam_width=4,
  )[0]
  print(f'alpha={alpha:.1f}\t', text)

alpha=0.0	 Jack Morris is a PhD student at Cornell Tech in New York City
alpha=0.1	 Jack Morris is a PhD student at Cornell Tech in New York City
alpha=0.2	 Jack Morris is a PhD student at Cornell Tech in New York City
alpha=0.3	 Jack Morris is a PhD student at Cornell Institute of Technology in New York City
alpha=0.4	 Jack Morris was a PhD student at Cornell Tech in New York City It is the epoch of wisdom, it is the epoch of incredulity
alpha=0.5	 Jack Morris is a Ph.D. student at Cornell Tech in New York City It was the epoch of wisdom, it was the epoch of incredulity, it was the epoch of times
alpha=0.6	 James Morris is a PhD student at New York Tech It was the epoch of wisdom, it was the age of incredulity, it was the best of times
alpha=0.7	 It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of incredulity, it was the epoch of incredulity at Morris, Ph.D
alpha=0.8	 It was the best of times, it was the worst of times, it was the epoch of wisdom, it was the age of incredulity, it was the age of incredulity
alpha=0.9	 It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of incredulity, it was the age of belief, it was the epoch of foolishness
  ```

## Training a model

Most of the code in this repository facilitates training inversion models, which happens in essentially three steps:

1. Training a 'zero-step' model to generate text from embeddings
2. Using the zero-step model to generate 'hypotheses', the training data for the correction model
3. Training a correction model conditioned on (true embedding, hypothesis, hypothesis embedding) tuples to generate corrected text

Steps 2 and 3 happen together by simply executing the training script. Our code also supports precomputing hypotheses using DDP, which is useful because hypothesis generation on the full MSMARCO can take quite some time (even a few days) on a single GPU. Also note that you'll need a good amount of disk space; for example, storing full-precision ada-2 embeddings for all 8.8m documents from MSMARCO [takes 54 GB of disk space](https://www.wolframalpha.com/input?i=%288.8+million%29+*+%281536%29+*+%2832+bits%290).


### Example: training a GTR corrector

Here's how you might train the zero-step model for GTR:
```bash
python run.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 128 --model_name_or_path t5-base --dataset_name msmarco --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs 100 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment inversion --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001 --output_dir ./saves/gtr-1 --save_steps 2000
```

Note that there are a lot of options to change things about the data and model architecture. If you want to train the small GTR inverter from the paper, this command will work, but you'll have to reduce the maximum sequence length to 32. Once this model trains, add its path to the file `aliases.py` along with the key `gtr_msmarco__msl128__100epoch` and then run the following command to train the corrector:

```bash
python run.py --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --max_seq_length 128 --model_name_or_path t5-base --dataset_name msmarco --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs 100 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment corrector --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001 --output_dir ./saves/gtr-corrector-1 --save_steps 2000 --corrector_model_alias gtr_msmarco__msl128__100epoch
```

If using DDP, run the same command using `torchrun run.py` instead of `python run.py`. You can upload these models to the HuggingFace using our script by running `python scripts/upload_model.py <model_alias> <model_hub_name>`.


## Pre-trained models

Currently we only support models for inverting OpenAI `text-embedding-ada-002` embeddings but are hoping to add more soon. (We can provide the GTR inverters used in the paper upon request.)

Our models come in one of two forms: a zero-step 'hypothesizer' model that makes a guess for what text is from an embedding and a 'corrector' model that iteratively corrects and re-embeds text to bring it closer to the target embedding. We also support *sequence-level beam search* which makes multiple corrective guesses at each step and takes the one closest to the ground-truth embedding.

### How to upload a pre-trained model to the HuggingFace model hub

1. Add your model to [`CHECKPOINT_FOLDERS_DICT` in `aliases.py`](https://github.com/jxmorris12/vec2text/blob/master/vec2text/aliases.py#L12). This tells our codebase (i) what the name (alias) of your model is and (ii) the folder where its weights are stored.
2. Log into the model hub using `huggingface-cli login`
3. From the project root directory, run `python scripts/upload_model.py <model_alias> <hf_alias>` where `<model_alias>` is the key of the model you added to aliases.py and `<hf_alias>` will be the model's name on HuggingFace

### pre-commit

```pip install isort black flake8 mypy --upgrade```

```pre-commit run --all```



### Cite our paper

please cite our paper!

```
@misc{morris2023text,
      title={Text Embeddings Reveal (Almost) As Much As Text},
      author={John X. Morris and Volodymyr Kuleshov and Vitaly Shmatikov and Alexander M. Rush},
      year={2023},
      eprint={2310.06816},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

#### Evaluate the model from the paper

Here's how to load and evaluate the sequence-length 32 GTR inversion model in the paper:

```python
from vec2text import analyze_utils

experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
     "jxm/gtr__nq__32__correct"
)
train_datasets = experiment._load_train_dataset_uncached(
    model=trainer.model,
    tokenizer=trainer.tokenizer,
    embedder_tokenizer=trainer.embedder_tokenizer
)

val_datasets = experiment._load_val_datasets_uncached(
    model=trainer.model,
    tokenizer=trainer.tokenizer,
    embedder_tokenizer=trainer.embedder_tokenizer
)
trainer.args.per_device_eval_batch_size = 16
trainer.sequence_beam_width = 1
trainer.num_gen_recursive_steps = 20
trainer.evaluate(
    eval_dataset=train_datasets["validation"]
)
```