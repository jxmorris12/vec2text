import analyze_utils
import models
import trainers


# TODO always load args from disk, delete this dict.
ARGS_DICT = {
    'dpr_corpus_msl32_beta': '--per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 32 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --exp_group_name mar17-baselines --learning_rate 0.0003 --freeze_strategy none --embedder_fake_with_zeros False --use_frozen_embeddings_as_input False --num_train_epochs 24 --max_eval_samples 500 --eval_steps 25000 --warmup_steps 100000 --bf16=1 --use_wandb=1'
}

# Dictionary mapping model names
CHECKPOINT_FOLDERS_DICT = {
    'dpr_corpus_msl32_beta': '/home/jxm3/research/retrieval/inversion/saves/db66b9c01b644541fedbdcc59c53a285/ebb31d91810c4b62d2b55b5382e8c7ea',
}


def load_inversion_trainer_from_alias(alias: str) -> trainers.InversionTrainer:
    args_str = ARGS_DICT[alias]
    checkpoint_folder = CHECKPOINT_FOLDERS_DICT[alias]
    trainer = analyze_utils.load_trainer(
        checkpoint_folder, args_str, do_eval=False
    )
    assert isinstance(trainer, trainers.InversionTrainer)
    return trainer


def load_inversion_model_from_alias(alias: str) -> models.InversionModel:
    trainer = load_inversion_trainer_from_alias(alias)
    assert isinstance(trainer.model, models.InversionModel)
    return trainer.model