import abc
import functools
import hashlib
import json
import logging
import os
import resource
import sys
from typing import Dict, Optional
import multiprocessing

import datasets
import torch
import transformers

import vec2text
from vec2text.collator import DataCollatorForCorrection
from vec2text.data_helpers import dataset_from_args, load_standard_val_datasets
from vec2text.models import (
    CorrectorEncoderFromLogitsModel,
    CorrectorEncoderModel,
    InversionFromLogitsEmbModel,
    InversionFromLogitsModel,
    InversionModel,
    InversionModelBagOfWords,
    InversionModelDecoderOnly,
    InversionModelNonAutoregressive,
)
from vec2text.models.config import InversionConfig
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments
from vec2text.tokenize_data import (
    embed_dataset_batch,
    tokenize_function,
    tokenize_function_llama_chat,
)
from vec2text.utils import MockEmbedder, dataset_map_multi_worker

# Allow W&B to start slowly.
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["_WANDB_STARTUP_DEBUG"] = "true"

# Don't send telemetry to HF every time we train.
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "False"
# os.environ["TOKENIZERS_PARALLELISM"] = "True"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger = logging.getLogger(__name__)

# We maintain our own cache because huggingface datasets caching
# doesn't always work properly.
DATASET_CACHE_PATH = os.environ.get(
    "VEC2TEXT_CACHE", os.path.expanduser("~/.cache/inversion")
)

# Noisy compilation from torch.compile
try:
    torch._logging.set_logs(dynamo=logging.INFO)
except AttributeError:
    # torch version too low
    pass


def md5_hash_kwargs(**kwargs) -> str:
    # We ignore special hf args that start with _ like '__cached__setup_devices'.
    safe_kwargs = {k: str(v) for k, v in kwargs.items() if not k.startswith("_")}
    s = json.dumps(safe_kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


def _get_num_proc(world_size) -> int:
    try:
        # NOTE: only available on some Unix platforms
        return (len(os.sched_getaffinity(0)) // world_size)  # type: ignore[attr-defined]
    except AttributeError:
        return (multiprocessing.cpu_count() // world_size)


class Experiment(abc.ABC):
    def __init__(
            self,
            model_args: ModelArguments,
            data_args: DataArguments,
            training_args: TrainingArguments,
    ):
        # Interactions between args handled here:
        training_args.metric_for_best_model = f"{data_args.dataset_name}_loss"

        logger.info(
            "Save checkpoints according to metric_for_best_model %s:",
            training_args.metric_for_best_model,
        )

        # Save all args.
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        # Set random seed, add hash to output path.
        transformers.set_seed(training_args.seed)

        if training_args.output_dir is None:
            training_args.output_dir = os.path.join("saves", self.kwargs_hash)
        print(f"Experiment output_dir = {training_args.output_dir}")
        # Set up output_dir and wandb.
        self._setup_logging()
        self._consider_init_wandb()

    @property
    def config(self) -> InversionConfig:
        return InversionConfig(
            **vars(self.data_args),
            **vars(self.model_args),
            **vars(self.training_args),
        )

    @property
    def is_llama_chat(self) -> bool:
        return self.model_args.embedder_model_name in [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
        ]

    @property
    def dataset_kwargs(self) -> Dict[str, str]:
        return {
            "model_name": self.model_args.model_name_or_path,
            "embedder_name": self.model_args.embedder_model_name,
            "max_seq_length": str(self.model_args.max_seq_length),
            "use_less_data": str(self.data_args.use_less_data),
            "embedder_model_api": str(self.model_args.embedder_model_api),
        }

    def _setup_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # if self.training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_error()

    def run(self):
        if self.training_args.do_eval:
            self.evaluate()
        else:
            self.train()

    def train(self) -> Dict:
        # *** Training ***
        training_args = self.training_args
        logger.info("*** Training ***")

        # Log on each process a small summary of training.
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"fp16 training: {training_args.fp16}, bf16 training: {training_args.bf16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")

        # Checkpointing logic
        checkpoint = self._get_checkpoint()
        logging.info("Experiment::train() loaded checkpoint %s", checkpoint)
        trainer = self.load_trainer()

        # Save model_args and data_args before training. Trainer will save training_args.
        if training_args.local_rank <= 0:
            torch.save(
                self.data_args, os.path.join(training_args.output_dir, "data_args.bin")
            )
            torch.save(
                self.model_args,
                os.path.join(training_args.output_dir, "model_args.bin"),
            )

        # train.   :)
        print(f"train() called – resume-from_checkpoint = {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        return metrics

    def evaluate(self) -> Dict:
        # *** Evaluation ***
        logger.info("*** Evaluate ***")
        trainer = self.load_trainer()
        num_eval_samples = len(trainer.eval_dataset)
        metrics = trainer.evaluate()
        max_eval_samples = (
            self.data_args.max_eval_samples
            if self.data_args.max_eval_samples is not None
            else num_eval_samples
        )
        metrics["eval_samples"] = min(max_eval_samples, num_eval_samples)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        return metrics

    def _get_checkpoint(self) -> Optional[str]:
        training_args = self.training_args
        last_checkpoint = None
        if (
                os.path.isdir(training_args.output_dir)
                and not training_args.overwrite_output_dir
        ):
            last_checkpoint = transformers.trainer_utils.get_last_checkpoint(
                training_args.output_dir
            )
            if (
                    last_checkpoint is None
                    and len(os.listdir(training_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                    last_checkpoint is not None
                    and training_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if checkpoint:
            logger.info("Loading from checkpoint %s", checkpoint)
        else:
            logger.info("No checkpoint found, training from scratch")

        return checkpoint

    @property
    def kwargs_hash(self) -> str:
        all_args = {
            **vars(self.model_args),
            **vars(self.data_args),
            **vars(self.training_args),
        }
        all_args.pop("local_rank")
        # print("all_args:", all_args)
        return md5_hash_kwargs(**all_args)

    @property
    def _world_size(self) -> int:
        try:
            return torch.distributed.get_world_size()
        except (RuntimeError, ValueError):
            return 1

    @property
    def _is_main_worker(self) -> bool:
        return (self.training_args.local_rank <= 0) and (
                int(os.environ.get("LOCAL_RANK", 0)) <= 0
        )

    @property
    @abc.abstractmethod
    def _wandb_project_name(self) -> str:
        raise NotImplementedError()

    @property
    def _wandb_exp_name(self) -> str:
        name_args = [
            self.training_args.exp_group_name,
            self.training_args.exp_name,
            self.model_args.model_name_or_path,
            self.model_args.embedder_model_name,
        ]
        name_args = [n for n in name_args if ((n is not None) and len(n))]
        return "__".join(name_args)

    def _consider_init_wandb(self) -> None:
        if self.training_args.use_wandb and self._is_main_worker:
            import wandb

            wandb.init(
                project=self._wandb_project_name,
                name=self._wandb_exp_name,
                id=self.kwargs_hash,
                resume=True,
            )
            wandb.config.update(
                {
                    **vars(self.model_args),
                    **vars(self.data_args),
                    **vars(self.training_args),
                },
                allow_val_change=True,
            )
            # Long-running experiments have been killed because wandb
            # runs out of file descriptors to write summary files
            # to. Very silly error, but seems unfixed:
            # https://github.com/wandb/wandb/issues/2825
            #
            # Anyway, this line of code should (hopefully) set the
            # limit to infinity so this can't happen.
            resource.setrlimit(
                resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
            )
        else:
            # Disable W&B
            pass
            # os.environ["WANDB_MODE"] = "disabled"
            # os.environ["WANDB_DISABLED"] = "true"

    @abc.abstractmethod
    def load_trainer(self) -> transformers.Trainer:
        raise NotImplementedError()

    @abc.abstractmethod
    def load_model(self) -> transformers.PreTrainedModel:
        raise NotImplementedError()

    def load_tokenizer(self) -> transformers.PreTrainedTokenizer:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            padding="max_length",
            truncation="max_length",
            max_length=self.model_args.max_seq_length,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Disable super annoying warning:
        # https://github.com/huggingface/transformers/issues/22638
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return tokenizer

    def get_collator(
            self, tokenizer: transformers.PreTrainedTokenizer
    ) -> transformers.DataCollatorForSeq2Seq:
        return transformers.DataCollatorForSeq2Seq(
            tokenizer,
            model=None,
            label_pad_token_id=-100,
            padding="max_length",
            max_length=self.model_args.max_seq_length,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

    def _load_train_dataset_uncached(
            self,
            model: transformers.PreTrainedModel,
            tokenizer: transformers.AutoTokenizer,
            embedder_tokenizer: transformers.AutoTokenizer,
    ) -> datasets.DatasetDict:
        data_args = self.data_args
        ###########################################################################
        # Load datasets
        logger.info("Loading dataset '%s'...", self.data_args.dataset_name)
        raw_datasets = dataset_from_args(self.data_args)

        # Remove extra features except for 'frozen_embeddings' which could be embeddings
        # saved to disk.
        column_names = list(raw_datasets["train"].features)
        ALLOWED_COLUMN_NAMES = {"frozen_embeddings"}
        column_names = [c for c in column_names if c not in ALLOWED_COLUMN_NAMES]

        # this argument allows us to *train* on less data (for example 1% of our training set).
        if data_args.use_less_data and (data_args.use_less_data > 0):
            for key in raw_datasets:
                new_length = min(len(raw_datasets[key]), data_args.use_less_data)
                raw_datasets[key] = raw_datasets[key].select(range(new_length))
        print(
            ">> using fast tokenizers:", tokenizer.is_fast, embedder_tokenizer.is_fast
        )

        tokenize_fn = (
            tokenize_function_llama_chat if self.is_llama_chat else tokenize_function
        )
        for key in raw_datasets:
            raw_datasets[key] = dataset_map_multi_worker(
                dataset=raw_datasets[key],
                map_fn=tokenize_fn(
                    tokenizer,
                    embedder_tokenizer,
                    "text",
                    self.model_args.max_seq_length,
                    padding=False,
                ),
                batched=True,
                num_proc=_get_num_proc(self._world_size),
                remove_columns=column_names,
                desc="Running tokenizer on dataset",
            )
        tokenized_datasets = raw_datasets
        ###########################################################################
        tokenized_datasets["train"].set_format("pt")
        tokenized_datasets["train"] = tokenized_datasets["train"].add_column(
            "idx", range(len(tokenized_datasets["train"]))
        )
        ###########################################################################
        if self.model_args.use_frozen_embeddings_as_input:
            print(
                f"[Precomputing embeddings with batch size: {self.training_args.per_device_train_batch_size}]"
            )
            assert torch.cuda.is_available()
            model = model.to(device)

            new_tokenized_datasets = {}
            for key, d in tokenized_datasets.items():
                new_fingerprint = (
                        d._fingerprint + md5_hash_kwargs(**self.dataset_kwargs) + ""
                )
                print("\tsaving precomputed embeddings to file:", new_fingerprint)
                new_tokenized_datasets[key] = dataset_map_multi_worker(
                    dataset=d,
                    map_fn=functools.partial(embed_dataset_batch, model),
                    batched=True,
                    batch_size=self.training_args.per_device_train_batch_size,
                    new_fingerprint=new_fingerprint,
                    num_proc=1,
                )
            tokenized_datasets = datasets.DatasetDict(new_tokenized_datasets)
        ###########################################################################
        max_eval_samples = min(
            len(tokenized_datasets["validation"]), self.data_args.max_eval_samples
        )
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(
            range(max_eval_samples)
        )
        tokenized_datasets["validation"] = tokenized_datasets["validation"].add_column(
            "idx", range(len(tokenized_datasets["validation"]))
        )
        tokenized_datasets["validation"].set_format("pt")
        ###########################################################################
        return tokenized_datasets

    def _prepare_val_datasets_dict(
            self,
            model: transformers.PreTrainedModel,
            tokenizer: transformers.AutoTokenizer,
            embedder_tokenizer: transformers.AutoTokenizer,
            val_datasets_dict: datasets.DatasetDict,
    ) -> datasets.DatasetDict:
        for name, dataset in val_datasets_dict.items():
            max_eval_samples = min(len(dataset), self.data_args.max_eval_samples)
            val_datasets_dict[name] = val_datasets_dict[name].select(
                range(max_eval_samples)
            )
            val_datasets_dict[name] = val_datasets_dict[name].add_column(
                "idx", range(len(val_datasets_dict[name]))
            )
            val_datasets_dict[name].set_format("pt")

        tokenize_fn = (
            tokenize_function_llama_chat if self.is_llama_chat else tokenize_function
        )
        for key in val_datasets_dict:
            val_datasets_dict[key] = dataset_map_multi_worker(
                dataset=val_datasets_dict[key],
                map_fn=tokenize_fn(
                    tokenizer=tokenizer,
                    embedder_tokenizer=embedder_tokenizer,
                    text_column_name="text",
                    max_seq_length=self.model_args.max_seq_length,
                    padding=False,
                ),
                remove_columns=["text"],
                batched=True,
                batch_size=1024,
                num_proc=_get_num_proc(self._world_size),
                desc="Running tokenizer on dataset",
            )

        # filter out empty examples (these exist for xsum documents).
        val_datasets_dict = val_datasets_dict.filter(lambda ex: ex["length"] > 1)

        if self.model_args.use_frozen_embeddings_as_input:
            assert torch.cuda.is_available()
            model = model.to(device)

            new_tokenized_datasets = {}
            for key, d in val_datasets_dict.items():
                new_tokenized_datasets[key] = dataset_map_multi_worker(
                    dataset=d,
                    map_fn=functools.partial(embed_dataset_batch, model),
                    batched=True,
                    batch_size=self.training_args.per_device_train_batch_size,
                    new_fingerprint=(
                            d._fingerprint + md5_hash_kwargs(**self.dataset_kwargs) + ""
                    ),
                    num_proc=1,
                )
            val_datasets_dict = datasets.DatasetDict(new_tokenized_datasets)
        return val_datasets_dict

    def _load_val_datasets_uncached(
            self,
            model: transformers.PreTrainedModel,
            tokenizer: transformers.AutoTokenizer,
            embedder_tokenizer: transformers.AutoTokenizer,
    ) -> datasets.DatasetDict:
        val_datasets_dict = load_standard_val_datasets()
        logger.info(
            "Loaded %d validation datasets: %s",
            len(val_datasets_dict),
            val_datasets_dict.keys(),
        )
        return self._prepare_val_datasets_dict(
            model=model,
            tokenizer=tokenizer,
            embedder_tokenizer=embedder_tokenizer,
            val_datasets_dict=val_datasets_dict,
        )

    def load_train_and_val_datasets(
            self,
            model: transformers.PreTrainedModel,
            tokenizer: transformers.AutoTokenizer,
            embedder_tokenizer: transformers.AutoTokenizer,
    ):
        dataset_kwargs: Dict[str, str] = self.dataset_kwargs

        # Only set this if it's true, for backwards-compatibility with
        # when we forgot to cache using this argument.
        if self.model_args.use_frozen_embeddings_as_input:
            dataset_kwargs["use_frozen_embeddings_as_input"] = "True"
            # Deprecated arg below. We used to cache different
            # embeddings for suffixes. Then they became the same.
            # Removing the below line will invalidate other
            # people's caches.
            dataset_kwargs["suffix_conditioning"] = "False"

        # os.environ["TOKENIZERS_PARALLELISM"] = "True"
        print(
            "Loading datasets with TOKENIZERS_PARALLELISM =",
            os.environ.get("TOKENIZERS_PARALLELISM"),
        )
        ######################################################################
        train_dataset_kwargs = {
            "dataset_name": self.data_args.dataset_name,
            **dataset_kwargs,
        }
        train_dataset_path = os.path.join(
            DATASET_CACHE_PATH, (md5_hash_kwargs(**train_dataset_kwargs) + ".arrow")
        )
        # Optionally set a train dataset path override
        train_dataset_path = os.environ.get(
            "VEC2TEXT_TRAIN_DATASET_PATH", train_dataset_path
        )
        if os.path.exists(train_dataset_path):
            print("loading train dataset from path:", train_dataset_path)
            train_datasets = datasets.load_from_disk(train_dataset_path)
        else:
            train_datasets = self._load_train_dataset_uncached(
                model=model,
                tokenizer=tokenizer,
                embedder_tokenizer=embedder_tokenizer,
            )
            print("saving train_dataset to path:", train_dataset_path)
            train_datasets.save_to_disk(
                train_dataset_path,
                max_shard_size="2GB",
            )
        ######################################################################
        val_dataset_kwargs = {
            "dataset_name": "__".join(
                ["ag_news", "arxiv", "xsum_doc", "xsum_summ", "wikibio"]
            ),
            **dataset_kwargs,
        }
        val_dataset_path = os.path.join(
            DATASET_CACHE_PATH, (md5_hash_kwargs(**val_dataset_kwargs) + ".arrow")
        )
        assert val_dataset_path != train_dataset_path
        if os.path.exists(val_dataset_path):
            val_datasets_dict = datasets.load_from_disk(val_dataset_path)
            print("loaded dict of val datasets from", val_dataset_path)
        else:
            val_datasets_dict = self._load_val_datasets_uncached(
                model=model,
                tokenizer=tokenizer,
                embedder_tokenizer=embedder_tokenizer,
            )
            print("saving val_dataset to path:", val_dataset_path)
            val_datasets_dict.save_to_disk(val_dataset_path)
        ######################################################################
        val_datasets_dict[self.data_args.dataset_name] = train_datasets["validation"]
        train_dataset = train_datasets["train"]

        for key in val_datasets_dict:
            new_length = min(
                len(val_datasets_dict[key]), self.data_args.max_eval_samples
            )
            val_datasets_dict[key] = val_datasets_dict[key].select(range(new_length))

        return (train_dataset, val_datasets_dict)


class InversionExperiment(Experiment):
    @property
    def trainer_cls(self):
        return vec2text.trainers.InversionTrainer

    @property
    def _wandb_project_name(self) -> str:
        return "emb-inv-4"

    def load_model(self) -> transformers.PreTrainedModel:
        return InversionModel(
            config=self.config,
        )

    def load_trainer(self) -> transformers.Trainer:
        model = self.load_model()
        train_dataset, eval_dataset = self.load_train_and_val_datasets(
            model=model,
            tokenizer=model.tokenizer,
            embedder_tokenizer=model.embedder_tokenizer,
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training model with name `{self.model_args.model_name_or_path}` - Total size={n_params / 2 ** 20:.2f}M params"
        )

        if self.training_args.mock_embedder:
            # This mode allows us to get the embedders off the GPU during training
            # once we've computed all the embeddings we need. :)
            assert (
                model.config.use_frozen_embeddings_as_input
            ), "must use frozen embeddings if mock_embedder=True"
            print(
                "IMPORTANT: Mocking embedder for the rest of training (to save GPU memory)."
                " Do not trust embedding-based evaluation metrics."
            )
            model.embedder.cpu()
            del model.embedder
            model.embedder = MockEmbedder(embedder_dim=model.embedder_dim)

        return self.trainer_cls(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.get_collator(tokenizer=model.tokenizer),
        )


class InversionFromLogitsExperiment(InversionExperiment):
    @property
    def trainer_cls(self):
        return vec2text.trainers.InversionFromLogitsTrainer

    @property
    def _wandb_project_name(self) -> str:
        return "emb-inv-logits-1"

    def load_model(self) -> transformers.PreTrainedModel:
        if self.training_args.experiment == "inversion_from_logits_emb":
            return InversionFromLogitsEmbModel(config=self.config)
        else:
            return InversionFromLogitsModel(
                config=self.config,
            )


class InversionExperimentDecoderOnly(InversionExperiment):
    def load_model(self) -> transformers.PreTrainedModel:
        return InversionModelDecoderOnly(
            config=self.config,
        )


class InversionExperimentNonAutoregressive(Experiment):
    @property
    def _wandb_project_name(self) -> str:
        return "emb-inv-na-1"

    def load_model(self) -> transformers.PreTrainedModel:
        return InversionModelNonAutoregressive(
            config=self.config,
        )

    def load_trainer(self) -> transformers.Trainer:
        model = self.load_model()
        train_dataset, eval_dataset = self.load_train_and_val_datasets(
            model=model,
            tokenizer=model.tokenizer,
            embedder_tokenizer=model.embedder_tokenizer,
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training model with name `{self.model_args.model_name_or_path}` - Total size={n_params / 2 ** 20:.2f}M params"
        )
        return vec2text.trainers.InversionTrainerNonAutoregressive(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.get_collator(tokenizer=model.tokenizer),
        )


class InversionExperimentBagOfWords(Experiment):
    @property
    def _wandb_project_name(self) -> str:
        return "emb-inv-bow-1"

    def load_model(self) -> transformers.PreTrainedModel:
        return InversionModelBagOfWords(
            config=self.config,
        )

    def load_trainer(self) -> transformers.Trainer:
        model = self.load_model()
        train_dataset, eval_dataset = self.load_train_and_val_datasets(
            model=model,
            tokenizer=model.tokenizer,
            embedder_tokenizer=model.embedder_tokenizer,
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training model with name `{self.model_args.model_name_or_path}` - Total size={n_params / 2 ** 20:.2f}M params"
        )
        return vec2text.trainers.InversionTrainerBagOfWords(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.get_collator(tokenizer=model.tokenizer),
        )


class CorrectorExperiment(Experiment):
    @property
    def _wandb_project_name(self) -> str:
        return "emb-correct-1"

    def load_trainer(self) -> transformers.Trainer:
        if self.training_args.corrector_model_from_pretrained:
            (
                _,
                inversion_trainer,
            ) = vec2text.analyze_utils.load_experiment_and_trainer_from_pretrained(
                name=self.training_args.corrector_model_from_pretrained,
                # max_seq_length=self.model_args.max_seq_length,
                use_less_data=self.data_args.use_less_data,
            )
        else:
            (
                _,
                inversion_trainer,
            ) = vec2text.aliases.load_experiment_and_trainer_from_alias(
                alias=self.training_args.corrector_model_alias,
                max_seq_length=self.model_args.max_seq_length,
                use_less_data=self.data_args.use_less_data,
            )
        model = self.load_model(inversion_trainer=inversion_trainer)
        return vec2text.trainers.Corrector(
            model=model,
            inversion_trainer=inversion_trainer,
            args=self.training_args,
            data_collator=DataCollatorForCorrection(
                tokenizer=inversion_trainer.model.tokenizer
            ),
        )

    def load_model(self, inversion_trainer) -> transformers.PreTrainedModel:
        exp = inversion_trainer.args.experiment
        if exp == "inversion_from_logits":
            self.config.embedder_dim = inversion_trainer.model.embedder_dim
            self.config.num_repeat_tokens = inversion_trainer.model.num_repeat_tokens
            return CorrectorEncoderFromLogitsModel(
                config=self.config,
            )
        else:
            return CorrectorEncoderModel(
                config=self.config,
            )


EXPERIMENT_CLS_MAP = {
    "inversion": InversionExperiment,
    "inversion_decoder_only": InversionExperimentDecoderOnly,
    "inversion_from_logits": InversionFromLogitsExperiment,
    "inversion_from_logits_emb": InversionFromLogitsExperiment,
    "corrector": CorrectorExperiment,
    "corrector_encoder": CorrectorExperiment,  # backwards-compatible; does same thing as just 'corrector'
    #
    "inversion_bow": InversionExperimentBagOfWords,
    "inversion_na": InversionExperimentNonAutoregressive,
}


def experiment_from_args(model_args, data_args, training_args) -> Experiment:
    if training_args.experiment in EXPERIMENT_CLS_MAP:
        experiment_cls = EXPERIMENT_CLS_MAP[training_args.experiment]  # type: ignore
    else:
        raise ValueError(f"Unknown experiment {training_args.experiment}")
    return experiment_cls(model_args, data_args, training_args)  # type: ignore
