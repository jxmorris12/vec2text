from typing import Optional, Tuple

import abc
import hashlib
import functools
import json
import logging
import os

import datasets
import torch
import transformers
from transformers import AutoTokenizer, set_seed

from collator import CustomCollator
from data_helpers import load_dpr_corpus, load_luar_reddit, NQ_DEV, NQ_TRAIN
from models import load_encoder_decoder, load_embedder_and_tokenizer, InversionModel
from run_args import ModelArguments, DataTrainingArguments, TrainingArguments
from tokenize_data import tokenize_function, whiten_embedded_dataset, embed_dataset_batch
from trainer import InversionTrainer


os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["_WANDB_STARTUP_DEBUG"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def md5_hash_kwargs(**kwargs) -> str:
    # We ignore special hf args that start with _ like '__cached__setup_devices'.
    safe_kwargs = {k: str(v) for k,v in kwargs.items() if not k.startswith('_')}
    s = json.dumps(safe_kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


class Experiment(abc.ABC):
    def __init__(model_args, data_args, training_args):
        set_seed(training_args.seed)
        # Add hash to output path.
        training_args.output_dir = os.path.join(
            training_args.output_dir, kwargs_hash        
        )
        # Save all args.
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        # Set up output_dir and wandb.
        self._setup_logging()
        self._consider_init_wandb()
    
    def _setup_logging() -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        if self.training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()
        log_level = self.training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    
    def run():
        if training_args.do_eval:
            self.evaluate()
        else:
            self.train()

    
    def train() -> Dict:
        # *** Training ***
        logger.info("*** Training ***")
        
        # Log on each process a small summary of training.
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")

        # Checkpointing logic
        checkpoint = self._get_checkpoint()

        # Save model_args and data_args before training. Trainer will save training_args.
        torch.save(data_args, os.path.join(training_args.output_dir, 'data_args.bin'))
        torch.save(model_args, os.path.join(training_args.output_dir, 'model_args.bin'))

        # train.
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        return metrics
    
    def evaluate() -> Dict:
        # *** Evaluation ***
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        return metrics
    
    def _get_checkpoint() -> Optional[str]:
        training_args = self.training_args
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        return checkpoint
    
    @property
    def kwargs_hash(self) -> str:
        return md5_hash_kwargs(**vars(model_args), **vars(data_args), **vars(training_args))
    
    @property
    def _is_main_worker() -> bool:
        return (self.training_args.local_rank <= 0) and (int(os.environ.get("LOCAL_RANK", 0)) <= 0)
    
    @property
    @abc.abstractmethod
    def _wandb_project_name() -> str:
        raise NotImplementedError()
    
    @property
    def _wandb_exp_name() -> str:
        name_args = (
            self.training_args.exp_group_name, 
            self.training_args.exp_name, 
            self.model_args.model_name_or_path, 
            self.model_args.embedder_model_name
        )
        name_args = [n for n in name_args if len(n)]
        return '__'.join(name_args)
        
    
    def _consider_init_wandb() -> None:
        if training_args.use_wandb and self._is_main_worker:
            import wandb
            wandb.init(
                project=self._wandb_project_name,
                name=self._wandb_exp_name,
                id=kwargs_hash,
                resume=True,
            )
            wandb.config.update(
                {**vars(self.model_args), **vars(self.data_args), **vars(self.training_args)},
                allow_val_change=True,
            )
        else:
            # Disable W&B
            os.environ["WANDB_MODE"] = "disabled"
            os.environ["WANDB_DISABLED"] = "true"

    @property
    @abc.abstractmethod
    def load_trainer(self) -> transformers.Trainer:
        raise NotImplementedError()
    
    @property
    @abc.abstractmethod
    def load_model(self) -> nn.Module:
        raise NotImplementedError()

    def load_train_and_val_datasets(self) -> Tuple[datasets.Dataset, datasets.Dataset]:
        ###########################################################################
        # Load datasets
        logger.info(f"Loading dataset '%s'...", self.data_args.dataset_name)
        raw_datasets = dataset_from_args(self.data_args)

        # Remove extra features except for 'frozen_embeddings' which could be embeddings
        # saved to disk.
        text_column_name = "text"
        column_names = list(raw_datasets["train"].features)
        ALLOWED_COLUMN_NAMES = { "frozen_embeddings" } # "document_id"}
        column_names = [c for c in column_names if c not in ALLOWED_COLUMN_NAMES]
        
        tokenized_datasets = raw_datasets.map(
            tokenize_function(tokenizer, embedder_tokenizer, text_column_name, model_args.max_seq_length),
            batched=True,
            # num_proc=training_args.dataloader_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

        ###########################################################################
        # Preprocess embeddings
        if model_args.use_frozen_embeddings_as_input:
            # files are just too big to cache :( 5 million 768-dim embeddings is 15gb 
            # datasets.disable_caching()
            # model = model.to(device)
            # tokenized_datasets = tokenized_datasets.map(
            #     functools.partial(embed_dataset_batch, model),
            #     batched=True,
            #     batch_size=training_args.per_device_train_batch_size,
            # )
            raise ValueError(f'broken feature - this breaks caching. fix caching to use.')

        if data_args.use_less_data:
            for key in tokenized_datasets:
                d = tokenized_datasets[key]
                new_length = max(256, int(len(d) * .1))
                tokenized_datasets[key] = tokenized_datasets[key].select(range(new_length))
        
        ###########################################################################
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]

    
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    return train_dataset, eval_dataset


class InversionExperiment(Experiment):

    @property
    def _wandb_project_name() -> str:
        return "emb-inv-1"
    
    @property
    def load_model(self) -> nn.Module:
        model_args = self.model_args
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            padding=True,
            truncation='max_length',
            max_length=model_args.max_seq_length,
        )
        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=model_args.embedder_model_name
        )
        return InversionModel(
            embedder=embedder,
            embedder_tokenizer=embedder_tokenizer,
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
            token_decode_alpha=model_args.token_decode_alpha,
            embeddings_from_layer_n=model_args.embeddings_from_layer_n,
        )
        
    @property
    def load_trainer(self) -> transformers.Trainer:
        model = self.load_model()
        train_dataset, eval_dataset = self.load_train_and_val_datasets()
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training model with name `{self.model_args.model_name_or_path}` - Total size={n_params/2**20:.2f}M params")
        raise InversionTrainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # tokenizer=model.tokenizer,
            data_collator=CustomCollator(tokenizer=model.tokenizer),
        )


class RerankingExperiment(Experiment):

    @property
    def _wandb_project_name() -> str:
        return "emb-rerank-1"

    @property
    def load_trainer(self) -> transformers.Trainer:
        raise InversionTrainer()
    
    @property
    def load_model(self) -> nn.Module:
        return ?


EXPERIMENT_CLS_MAP = {
    'inversion': InversionExperiment,
    'reranking': RerankingExperiment,
}
def setup_experiment(model_args, data_args, training_args) -> Experiment
    if training_args.experiment in EXPERIMENT_CLS_MAP:
        experiment_cls = EXPERIMENT_CLS_MAP[training_args.experiment]
    else:
        raise ValueError(f'Unknown experiment {training_args.experiment}')
    return experiment_cls(model_args, data_args, training_args)
