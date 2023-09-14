import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

from vec2text.models import (
    EMBEDDER_MODEL_NAMES,
    EMBEDDING_TRANSFORM_STRATEGIES,
    FREEZE_STRATEGIES,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DATASET_NAMES = ["nq", "luar_reddit", "msmarco"]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        ###
        ## huggingface.co/facebook/dpr-ctx_encoder-single-nq-base
        ###
        default="t5-base",
        metadata={
            "help": (
                "The model checkpoint for weights initialization .Don't set if you want to train a model from scratch."
            )
        },
    )
    embedder_model_name: str = field(
        ###
        ## huggingface.co/facebook/dpr-ctx_encoder-single-nq-base
        ###
        default="gtr_base",
        metadata={
            "help": "Model to get embeddings from (locally)",
            "choices": EMBEDDER_MODEL_NAMES,
        },
    )
    embedder_model_api: Optional[str] = field(
        default=None, metadata={"help": "API to get embeddings from"}
    )
    embedding_transform_strategy: str = field(
        default="repeat",
        metadata={
            "help": "Strategy for transforming from sentence embedding into sequence-level input for encoder-decoder",
            "choices": EMBEDDING_TRANSFORM_STRATEGIES,
        },
    )
    encoder_dropout_disabled: bool = field(
        default=False, metadata={"help": "Disable dropout on T5 encoder"}
    )
    decoder_dropout_disabled: bool = field(
        default=False, metadata={"help": "Disable dropout on T5 decoder"}
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    max_seq_length: int = field(
        default=128, metadata={"help": "Maximum sequence length for tokenizer"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    num_repeat_tokens: int = field(
        default=16,
        metadata={
            "help": "Number of times to repeat embedding along T5 input sequence length."
        },
    )
    embedder_no_grad: bool = field(
        default=True, metadata={"help": "Whether to disable grads for DPR"}
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Whether to use LORA+int8 for fine-tuning"}
    )
    embedder_fake_with_zeros: bool = field(
        default=False,
        metadata={
            "help": "Whether to pass all zeros as embedding (and not use DPR at all)"
        },
    )
    use_frozen_embeddings_as_input: bool = field(
        default=False,
        metadata={
            "help": "Whether to pass a 'frozen_embedding' column and train on that instead of generating embeddings on-the-fly"
        },
    )
    corrector_ignore_hypothesis_embedding: bool = field(
        default=False,
        metadata={
            "help": "If set, and training corrector encoder, will ignore the hypothesis embedding"
        },
    )
    embeddings_from_layer_n: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, uses embeddings from layer n - for example set to 0 to use word embeddings"
        },
    )
    freeze_strategy: str = field(
        default="none",
        metadata={
            "help": "which part of the model to freeze",
            "choices": FREEZE_STRATEGIES,
        },
    )
    suffix_conditioning: bool = field(
        default=False,
        metadata={"help": "for logits inversion, whether to condition on the suffix"},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="msmarco",
        metadata={
            "choices": DATASET_NAMES,
            "help": "The name of the dataset to use (via the datasets library).",
        },
    )
    max_eval_samples: int = field(
        default=1000,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    use_less_data: int = field(
        default=-1,
        metadata={
            "help": {"Use a small amount of the training/eval data (for testing)"}
        },
    )

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Need a dataset name.")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # https://github.com/huggingface/transformers/blob/e82c1cb78e178519060b9391214727be75a218ca/src/transformers/training_args.py#L121
    output_dir: str = field(
        default="saves", metadata={"help": "Output directory for training saves"}
    )
    corrector_model_alias: str = field(
        default="dpr_nq__msl32_beta",
        metadata={"help": "Alias of corrector model to train (defined in aliases.py)"},
    )
    cheat_on_train_hypotheses: bool = field(
        default=False,
        metadata={
            "help": "When set, will interpolate true with pred train hypothesis for 'closer' training data"
        },
    )

    steps_per_epoch: int = field(
        default=500_000,
        metadata={"required": False, "help": "Size of pseudo-training set."},
    )
    num_train_epochs: float = field(
        default=30.0,
        metadata={"required": False, "help": "Number of epochs for training"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW on the backbone model."},
    )
    use_wandb: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to log to Weights & Biases."}
    )
    report_to: str = "wandb"
    per_device_train_batch_size: int = field(
        default=128, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": ("Whether to use bf16 (mixed) precision instead of 32-bit.")},
    )
    # torch_compile: bool = True # for torch 2

    ##################### Experimental Settings ####################
    experiment: str = field(
        default="inversion",
        metadata={
            "required": False,
            "help": "Which experiment to run (defines model, loss func, dataset...) ",
            "choices": [
                "inversion",  # our model: projects and feeds to encoder-decoder
                "inversion_from_logits",
                "inversion_decoder_only",  # baseline: use single embedding as input to a decoder
                "inversion_bow",
                "inversion_na",
                "reranking",
                "corrector",
                "corrector_encoder",
            ],
        },
    )
    exp_name: str = field(
        default="",
        metadata={
            "required": False,
            "help": "Name to identify this specific run of an experiment",
        },
    )
    exp_group_name: str = field(
        default="",
        metadata={
            "required": False,
            "help": "Name to identify this sweep / series of experiments",
        },
    )

    # Need to *not* remove unused columns so we keep query_attention_mask, etc.
    # which huggingface doesn't think we need.
    remove_unused_columns: bool = False

    # Do evaluation and logging on certain num steps.
    evaluation_strategy: str = "steps"
    logging_strategy: str = "steps"
    save_strategy: str = "steps"

    save_total_limit: int = 1  # Maximum number of checkpoints to save.

    warmup_steps: int = field(
        default=4000, metadata={"help": "Number of steps of warmup"}
    )
    logging_steps: int = field(
        default=400, metadata={"help": "Number of steps between logging metrics"}
    )
    save_steps: int = 8000
    eval_steps: int = field(
        default=40000,
        metadata={
            "help": "Number of steps between eval (will be scaled as if batch size is 32)"
        },
    )

    include_inputs_for_metrics: bool = True

    def __setattr__(self, name, value):
        super(transformers.TrainingArguments, self).__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()
        self._frozen = True
        self.report_to = (
            ["wandb"] if (self.use_wandb and (self.local_rank <= 0)) else []
        )
        self.dataloader_pin_memory = True
        num_workers = int(len(os.sched_getaffinity(0)) / torch.cuda.device_count())
        os.environ["RAYON_RS_NUM_CPUS"] = str(
            num_workers
        )  # Sets threads for hf tokenizers
        self.dataloader_num_workers = num_workers
        print(f"Set num workers to {num_workers}")

        self.dataloader_drop_last = False

        # Scale logging steps proportional to batch size.
        self.warmup_steps = round(self.warmup_steps * (32 / self.train_batch_size))
        self.logging_steps = round(self.logging_steps * (32 / self.train_batch_size))
        self.eval_steps = round(self.eval_steps * (32 / self.train_batch_size))
        self.save_steps = round(self.save_steps * (32 / self.train_batch_size))

        # defaults from SentenceTransformers
        # lr 2e-5
        self.adam_epsilon = 1e-6

        self.group_by_length = True
        self.length_column_name = "length"

        self.load_best_model_at_end = True
        self.greater_is_better = False

        self.do_eval = False
        # self.ddp_backend = "gloo"
