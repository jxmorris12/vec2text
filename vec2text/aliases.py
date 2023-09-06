from vec2text import analyze_utils

# TODO always load args from disk, delete this dict.
ARGS_DICT = {
    "dpr_nq__msl32_beta": "--dataset_name nq --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 32 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --exp_group_name mar17-baselines --learning_rate 0.0003 --freeze_strategy none --embedder_fake_with_zeros False --use_frozen_embeddings_as_input False --num_train_epochs 24 --max_eval_samples 500 --eval_steps 25000 --warmup_steps 100000 --bf16=1 --use_wandb=0",
    "gtr_nq__msl128_beta": "--dataset_name nq --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 128 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --exp_group_name mar17-baselines --learning_rate 0.0003 --freeze_strategy none --embedder_fake_with_zeros False --use_frozen_embeddings_as_input False --num_train_epochs 24 --max_eval_samples 500 --eval_steps 25000 --warmup_steps 100000 --bf16=1 --use_wandb=0",
    # "gtr_nq__msl32_beta__correct": "--experiment corrector_encoder --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --max_seq_length 32 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --exp_group_name may19-corr-encoder --learning_rate 0.002 --freeze_strategy none --embedder_fake_with_zeros False --use_frozen_embeddings_as_input False --encoder_dropout_disabled False --decoder_dropout_disabled False --use_less_data -1 --num_train_epochs 60 --max_eval_samples 500 --eval_steps 25000 --warmup_steps 200000 --bf16=1 --use_lora=0 --use_wandb=1",
    # "openai_msmarco__msl128__100epoch": "--per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 128 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --learning_rate 0.0002 --freeze_strategy none --embedder_fake_with_zeros False --encoder_dropout_disabled False --decoder_dropout_disabled False --use_less_data 1000000 --num_train_epochs 100 --max_eval_samples 500 --eval_steps 50000 --warmup_steps 20000 --bf16=1 --use_lora=0 --use_wandb=0 --embedder_model_api text-embedding-ada-002 --use_frozen_embeddings_as_input True --exp_group_name jun3-openai-4gpu-ddp-3",
}

# Dictionary mapping model names
CHECKPOINT_FOLDERS_DICT = {
    ####################################################################
    ######################## Natural Questions #########################
    ####################################################################
    #  https://wandb.ai/jack-morris/emb-inv-1/runs/ebb31d91810c4b62d2b55b5382e8c7ea/logs?workspace=user-jxmorris12
    #  (This should be called GTR, not DPR; misnomer retained for legacy purposes.) [loss 1.04]
    "dpr_nq__msl32_beta": "/home/jxm3/research/retrieval/inversion/saves/db66b9c01b644541fedbdcc59c53a285/ebb31d91810c4b62d2b55b5382e8c7ea",
    #  https://wandb.ai/jack-morris/emb-inv-1/runs/dc72e8b9c01bd27b0ed1c2def90bcee5/overview?workspace=user-jxmorris12
    #   (achieves BLEU of 11.7 w/ no_gram_repeats=3)
    "gtr_nq__msl128_beta": "/home/jxm3/research/retrieval/inversion/saves/8631b1c05efebde3077d16c5b99f6d5e/dc72e8b9c01bd27b0ed1c2def90bcee5",
    #  https://wandb.ai/jack-morris/emb-correct-1/runs/e9430bc73cfd6fb433eb0e5401d4a7ff [loss .644]
    "gtr_nq__msl32_beta__correct": "/home/jxm3/research/retrieval/inversion/saves/47d9c149a8e827d0609abbeefdfd89ac",
    # https://wandb.ai/jack-morris/emb-correct-1/runs/aa389ad434c01b692df796c2e2eb599c?workspace=user-jxmorris12 [loss .82]
    "gtr_nq__msl32_beta__correct__nofeedback": "/home/jxm3/research/retrieval/inversion/saves/f031f2b69c815cca265dd791473de60a",
    # https://wandb.ai/jack-morris/emb-inv-3/runs/35f5983783e5d6c7613aa14b84af1ff6/overview?workspace=
    "clinicalbert_nq__msl32": "/home/jxm3/research/retrieval/inversion/saves/01c63decd9009f5961504b52a96cd324",
    ####################################################################
    ############################# MSMARCO ##############################
    ####################################################################
    # gtr hypothesis model (60 epochs trained) [loss 2.04, bleu ~12.7]:
    #   https://wandb.ai/jack-morris/emb-inv-3/runs/d8319570c0314d95b2a9746f849e6218/overview?workspace=user-jxmorris12
    "gtr_msmarco__msl128__100epoch": "/home/jxm3/research/retrieval/inversion/saves/d6312870a6f49dee914198d048ee88f4",
    # openai hypothesis model [sl32] [still training, loss 1.233, bleu 29...] https://wandb.ai/jack-morris/emb-inv-3/runs/9b5d4aac9b16dad6d4a8c65cbc1a8859?workspace=user-jxmorris12
    "openai_msmarco__msl32__100epoch": "/home/jxm3/research/retrieval/inversion/saves/61becf9bb1d627272cd1923ac4871e73",
    # openai corrector model [sl32] [loss 0.77, bleu 45]
    "openai_msmarco__msl32__100epoch__correct": "/home/jxm3/research/retrieval/inversion/saves/7758f43e621db8ee718306f31139e3b0",
    # openai hypothesis model [loss 1.8, bleu ~14.5]:
    #    https://wandb.ai/jack-morris/emb-inv-3/runs/4dc5011fd9be6b1f4dd3f7f4aa351165?workspace=user-jxmorris12
    "openai_msmarco__msl128__100epoch": "/home/jxm3/research/retrieval/inversion/saves/f9abd65db4c4823264b133816d08612f",
    # openai corrector model:
    # https://wandb.ai/jack-morris/emb-correct-1/runs/b3b83aede945ba412ac6e9eebaf5f0dd/overview?workspace=user-jxmorris12
    "openai_msmarco__msl128__100epoch__correct": "/home/jxm3/research/retrieval/inversion/saves/d6ec9d5838a4ad3daeba636e5378a8a0",
    # openai corrector model trained for a lot longer:
    # https://wandb.ai/jack-morris/emb-correct-1/runs/5c1f59956a46e62e8f26f778e167348a/overview?workspace=user-jxmorris12
    "openai_msmarco__msl128__200epoch__correct": "/home/jxm3/research/retrieval/inversion/saves/c7be16d4af952eea8046a02a9d2a2113",
}


def load_experiment_and_trainer_from_alias(
    alias: str, max_seq_length: int = None, use_less_data: int = None
):  # -> trainers.InversionTrainer:
    args_str = ARGS_DICT.get(alias)
    checkpoint_folder = CHECKPOINT_FOLDERS_DICT[alias]
    print(f"loading alias {alias} from {checkpoint_folder}...")
    experiment, trainer = analyze_utils.load_experiment_and_trainer(
        checkpoint_folder,
        args_str,
        do_eval=False,
        max_seq_length=max_seq_length,
        use_less_data=use_less_data,
    )
    return experiment, trainer


def load_model_from_alias(alias: str, max_seq_length: int = None):
    _, trainer = load_experiment_and_trainer_from_alias(
        alias, max_seq_length=max_seq_length
    )
    return trainer.model
