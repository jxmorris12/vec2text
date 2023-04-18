import abc
import os

import transformers


class Experiment(abc.ABC):
    def __init__(training_args, model_args, data_args):
        set_seed(training_args.seed)
        # Add hash to output path.
        training_args.output_dir = os.path.join(
            training_args.output_dir, kwargs_hash        
        )
        # Save all args.
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        # Set up output_dir and wandb.
        self._consider_init_wandb()
    
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
        raise InversionTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=CustomCollator(tokenizer=tokenizer),
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