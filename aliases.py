import analyze_utils

# TODO always load args from disk, delete this dict.
ARGS_DICT = {
    "dpr_nq__msl32_beta": "--per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 32 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --exp_group_name mar17-baselines --learning_rate 0.0003 --freeze_strategy none --embedder_fake_with_zeros False --use_frozen_embeddings_as_input False --num_train_epochs 24 --max_eval_samples 500 --eval_steps 25000 --warmup_steps 100000 --bf16=1 --use_wandb=0",
    "gtr_nq__msl128_beta": "--per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 128 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --exp_group_name mar17-baselines --learning_rate 0.0003 --freeze_strategy none --embedder_fake_with_zeros False --use_frozen_embeddings_as_input False --num_train_epochs 24 --max_eval_samples 500 --eval_steps 25000 --warmup_steps 100000 --bf16=1 --use_wandb=0",
    # "openai_msmarco__msl128__100epoch": "--per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 128 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --learning_rate 0.0002 --freeze_strategy none --embedder_fake_with_zeros False --encoder_dropout_disabled False --decoder_dropout_disabled False --use_less_data 1000000 --num_train_epochs 100 --max_eval_samples 500 --eval_steps 50000 --warmup_steps 20000 --bf16=1 --use_lora=0 --use_wandb=0 --embedder_model_api text-embedding-ada-002 --use_frozen_embeddings_as_input True --exp_group_name jun3-openai-4gpu-ddp-3",
}

# Dictionary mapping model names
CHECKPOINT_FOLDERS_DICT = {
    ####################################################################
    ######################## Natural Questions #########################
    ####################################################################
    #  https://wandb.ai/jack-morris/emb-inv-1/runs/ebb31d91810c4b62d2b55b5382e8c7ea/logs?workspace=user-jxmorris12
    #  (This should be called GTR, not DPR, retained for legacy purposes.)
    "dpr_nq__msl32_beta": "/home/jxm3/research/retrieval/inversion/saves/db66b9c01b644541fedbdcc59c53a285/ebb31d91810c4b62d2b55b5382e8c7ea",
    #  https://wandb.ai/jack-morris/emb-inv-1/runs/dc72e8b9c01bd27b0ed1c2def90bcee5/overview?workspace=user-jxmorris12
    #   (achieves BLEU of 11.7 w/ no_gram_repeats=3)
    "gtr_nq__msl128_beta": "/home/jxm3/research/retrieval/inversion/saves/8631b1c05efebde3077d16c5b99f6d5e/dc72e8b9c01bd27b0ed1c2def90bcee5",
    #  https://wandb.ai/jack-morris/emb-correct-1/runs/e9430bc73cfd6fb433eb0e5401d4a7ff
    "gtr_nq__msl32_beta__correct": "/home/jxm3/research/retrieval/inversion/saves/47d9c149a8e827d0609abbeefdfd89ac",
    ####################################################################
    ############################# MSMARCO ##############################
    ####################################################################
    # https://wandb.ai/jack-morris/emb-inv-3/runs/4dc5011fd9be6b1f4dd3f7f4aa351165?workspace=user-jxmorris12
    "openai_msmarco__msl128__100epoch": "/home/jxm3/research/retrieval/inversion/saves/f9abd65db4c4823264b133816d08612f",
}


def load_inversion_trainer_from_alias(alias: str):  # -> trainers.InversionTrainer:
    import trainers

    args_str = ARGS_DICT.get(alias)
    checkpoint_folder = CHECKPOINT_FOLDERS_DICT[alias]
    print(f"loading alias {alias} from {checkpoint_folder}...")
    trainer = analyze_utils.load_trainer(checkpoint_folder, args_str, do_eval=False)
    assert isinstance(trainer, trainers.InversionTrainer)
    return trainer


def load_inversion_model_from_alias(alias: str):  # -> models.InversionModel:
    import models

    trainer = load_inversion_trainer_from_alias(alias)
    assert isinstance(trainer.model, models.InversionModel)
    return trainer.model
