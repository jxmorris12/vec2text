import abc
import functools
import hashlib
import json
import logging
import os
import resource
import sys
from typing import Dict, Optional

import datasets
import torch
import transformers

import aliases
import trainers
from data_helpers import dataset_from_args, load_standard_val_datasets
from models import (
    CorrectorEncoderModel,
    CorrectorModel,
    InversionModel,
    InversionModelBagOfWords,
    InversionModelNonAutoregressive,
    load_embedder_and_tokenizer,
    load_encoder_decoder,
)
from run_args import DataArguments, ModelArguments, TrainingArguments
from tokenize_data import embed_dataset_batch, tokenize_function
from utils import torch_main_worker_finish_first

# Allow W&B to start slowly.
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["_WANDB_STARTUP_DEBUG"] = "true"

# For batch decoding outputs during evaluation.
os.environ["TOKENIZERS_PARALLELISM"] = "True"

# Don't send telemetry to HF every time we train.
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


# We maintain our own cache because huggingface datasets caching
# doesn't work properly.
DATASET_CACHE_PATH = "/home/jxm3/.cache/inversion"


def md5_hash_kwargs(**kwargs) -> str:
    # We ignore special hf args that start with _ like '__cached__setup_devices'.
    safe_kwargs = {k: str(v) for k, v in kwargs.items() if not k.startswith("_")}
    s = json.dumps(safe_kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


class Experiment(abc.ABC):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        # Save all args.
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        # Add hash to output path.
        transformers.set_seed(training_args.seed)
        training_args.output_dir = os.path.join(
            training_args.output_dir, self.kwargs_hash
        )
        # Set up output_dir and wandb.
        self._setup_logging()
        self._consider_init_wandb()

    def _setup_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # if self.training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        # transformers.utils.logging.set_verbosity_info()
        log_level = self.training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

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
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
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
            # and not training_args.overwrite_output_dir
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
        return md5_hash_kwargs(
            **vars(self.model_args), **vars(self.data_args), **vars(self.training_args)
        )

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

    @property
    @abc.abstractmethod
    def load_trainer(self) -> transformers.Trainer:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def load_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    def get_collator(
        self, tokenizer: transformers.PreTrainedTokenizer
    ) -> transformers.DataCollatorForSeq2Seq:
        return transformers.DataCollatorForSeq2Seq(
            tokenizer,
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

    @torch_main_worker_finish_first
    def _load_train_dataset_uncached(
        self,
        model: torch.nn.Module,
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
        ALLOWED_COLUMN_NAMES = {"frozen_embeddings"}  # "document_id"}
        column_names = [c for c in column_names if c not in ALLOWED_COLUMN_NAMES]

        # this argument allows us to *train* on less data (1% of our training set).
        if data_args.use_less_data and (data_args.use_less_data > 0):
            for key in raw_datasets:
                new_length = min(len(raw_datasets[key]), data_args.use_less_data)
                raw_datasets[key] = raw_datasets[key].select(range(new_length))

        tokenized_datasets = raw_datasets.map(
            tokenize_function(
                tokenizer,
                embedder_tokenizer,
                "text",
                self.model_args.max_seq_length,
                padding=False,
            ),
            batched=True,
            # num_proc=training_args.dataloader_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

        ###########################################################################
        tokenized_datasets["train"].set_format("pt")
        tokenized_datasets["train"] = tokenized_datasets["train"].add_column(
            "idx", range(len(tokenized_datasets["train"]))
        )
        ###########################################################################
        if self.model_args.use_frozen_embeddings_as_input:
            tokenized_datasets = tokenized_datasets.map(
                functools.partial(embed_dataset_batch, model),
                batched=True,
                batch_size=self.training_args.per_device_train_batch_size,
            )
        ###########################################################################
        return tokenized_datasets

    def _load_val_datasets_uncached(
        self,
        model: torch.nn.Module,
        tokenizer: transformers.AutoTokenizer,
        embedder_tokenizer: transformers.AutoTokenizer,
    ) -> datasets.DatasetDict:
        data_args = self.data_args
        val_datasets_dict = load_standard_val_datasets()
        logger.info(
            "Loaded %d validation datasets: %s",
            len(val_datasets_dict),
            val_datasets_dict.keys(),
        )

        for name, dataset in val_datasets_dict.items():
            max_eval_samples = min(len(dataset), data_args.max_eval_samples)
            val_datasets_dict[name] = val_datasets_dict[name].select(
                range(max_eval_samples)
            )
            val_datasets_dict[name] = val_datasets_dict[name].add_column(
                "idx", range(len(val_datasets_dict[name]))
            )
            val_datasets_dict[name].set_format("pt")

        val_datasets_dict = val_datasets_dict.map(
            tokenize_function(
                tokenizer,
                embedder_tokenizer,
                "text",
                self.model_args.max_seq_length,
            ),
            remove_columns=["text"],
            batched=True,
            desc="Running tokenizer on dataset",
        )

        if self.model_args.use_frozen_embeddings_as_input:
            val_datasets_dict = val_datasets_dict.map(
                functools.partial(embed_dataset_batch, model),
                batched=True,
                batch_size=self.training_args.per_device_train_batch_size,
            )

        # filter out empty examples (these exist for xsum documents).
        val_datasets_dict = val_datasets_dict.filter(lambda ex: ex["length"] > 1)
        return val_datasets_dict

    def load_train_and_val_datasets(
        self,
        model: torch.nn.Module,
        tokenizer: transformers.AutoTokenizer,
        embedder_tokenizer: transformers.AutoTokenizer,
    ):
        dataset_kwargs = {
            "model_name": self.model_args.model_name_or_path,
            "embedder_name": self.model_args.embedder_model_name,
            "max_seq_length": self.model_args.max_seq_length,
            "use_less_data": self.data_args.use_less_data,
        }
        ######################################################################
        train_dataset_kwargs = {
            "dataset_name": self.data_args.dataset_name,
            **dataset_kwargs,
        }
        train_dataset_path = os.path.join(
            DATASET_CACHE_PATH, (md5_hash_kwargs(**train_dataset_kwargs) + ".arrow")
        )
        if os.path.exists(train_dataset_path):
            train_datasets = datasets.load_from_disk(train_dataset_path)
        else:
            train_datasets = self._load_train_dataset_uncached(
                model=model,
                tokenizer=tokenizer,
                embedder_tokenizer=embedder_tokenizer,
            )
            train_datasets.save_to_disk(train_dataset_path)
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
        else:
            val_datasets_dict = self._load_val_datasets_uncached(
                model=model,
                tokenizer=tokenizer,
                embedder_tokenizer=embedder_tokenizer,
            )
            val_datasets_dict.save_to_disk(val_dataset_path)
        ######################################################################
        train_dataset = train_datasets["train"]
        val_datasets_dict[self.data_args.dataset_name] = train_datasets["validation"]

        return (train_dataset, val_datasets_dict)


class InversionExperiment(Experiment):
    @property
    def _wandb_project_name(self) -> str:
        return "emb-inv-3"

    def load_model(self) -> torch.nn.Module:
        model_args = self.model_args
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            padding=True,
            truncation="max_length",
            max_length=model_args.max_seq_length,
        )
        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=model_args.embedder_model_name
        )
        return InversionModel(
            embedder=embedder,
            embedder_tokenizer=embedder_tokenizer,
            embedder_model_api=model_args.embedder_model_api,
            tokenizer=tokenizer,
            encoder_decoder=load_encoder_decoder(
                model_name=model_args.model_name_or_path,
                lora=model_args.use_lora,
            ),
            num_repeat_tokens=model_args.num_repeat_tokens,
            embedder_no_grad=model_args.embedder_no_grad,
            embedder_fake_with_zeros=model_args.embedder_fake_with_zeros,
            use_frozen_embeddings_as_input=model_args.use_frozen_embeddings_as_input,
            whiten_embeddings=model_args.whiten_embeddings,
            encoder_dropout_disabled=model_args.encoder_dropout_disabled,
            decoder_dropout_disabled=model_args.decoder_dropout_disabled,
            freeze_strategy=model_args.freeze_strategy,
            encoder_decoder_lora=model_args.use_lora,
            embeddings_from_layer_n=model_args.embeddings_from_layer_n,
        )

    @torch_main_worker_finish_first
    def load_trainer(self) -> transformers.Trainer:
        model = self.load_model()
        train_dataset, eval_dataset = self.load_train_and_val_datasets(
            model=model,
            tokenizer=model.tokenizer,
            embedder_tokenizer=model.embedder_tokenizer,
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training model with name `{self.model_args.model_name_or_path}` - Total size={n_params/2**20:.2f}M params"
        )

        return trainers.InversionTrainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.get_collator(tokenizer=model.tokenizer),
        )


class InversionExperimentNonAutoregressive(Experiment):
    @property
    def _wandb_project_name(self) -> str:
        return "emb-inv-na-1"

    def load_model(self) -> torch.nn.Module:
        model_args = self.model_args
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            padding=True,
            truncation="max_length",
            max_length=model_args.max_seq_length,
        )
        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=model_args.embedder_model_name
        )
        encoder = transformers.AutoModel.from_pretrained(
            model_args.model_name_or_path,
        ).encoder
        return InversionModelNonAutoregressive(
            embedder=embedder,
            encoder=encoder,
            embedder_tokenizer=embedder_tokenizer,
            tokenizer=tokenizer,
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
            f"Training model with name `{self.model_args.model_name_or_path}` - Total size={n_params/2**20:.2f}M params"
        )
        return trainers.InversionTrainerNonAutoregressive(
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

    def load_model(self) -> torch.nn.Module:
        model_args = self.model_args
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            padding=True,
            truncation="max_length",
            max_length=model_args.max_seq_length,
        )
        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=model_args.embedder_model_name
        )
        encoder = transformers.AutoModel.from_pretrained(
            model_args.model_name_or_path,
        ).encoder
        return InversionModelBagOfWords(
            embedder=embedder,
            encoder=encoder,
            embedder_tokenizer=embedder_tokenizer,
            tokenizer=tokenizer,
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
            f"Training model with name `{self.model_args.model_name_or_path}` - Total size={n_params/2**20:.2f}M params"
        )
        return trainers.InversionTrainerBagOfWords(
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
        # TODO: argparse for this
        inversion_trainer = aliases.load_inversion_trainer_from_alias(
            alias=self.training_args.corrector_model_alias,
        )
        return trainers.CorrectorTrainer(
            model=self.load_model(),
            inversion_trainer=inversion_trainer,
            args=self.training_args,
        )

    def load_model(self) -> torch.nn.Module:
        raise RuntimeError(
            "Did you mean to launch the CorrectorEncoder experiment instead?"
        )
        encoder_decoder = transformers.AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        return CorrectorModel(encoder_decoder=encoder_decoder)


class CorrectorEncoderExperiment(CorrectorExperiment):
    def load_model(self) -> torch.nn.Module:
        encoder_decoder = transformers.AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        return CorrectorEncoderModel(
            encoder_decoder=encoder_decoder,
            ignore_hypothesis_embedding=self.model_args.corrector_ignore_hypothesis_embedding,
        )


EXPERIMENT_CLS_MAP = {
    "inversion": InversionExperiment,
    "corrector": CorrectorExperiment,
    "corrector_encoder": CorrectorEncoderExperiment,
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
