import pytest

from vec2text import analyze_utils


def test_load_model_seqlen32():
    checkpoint_folder = "/home/jxm3/research/retrieval/inversion/saves/db66b9c01b644541fedbdcc59c53a285/ebb31d91810c4b62d2b55b5382e8c7ea"
    args_str = "--per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 32 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --exp_group_name mar17-baselines --learning_rate 0.0003 --freeze_strategy none --embedder_fake_with_zeros False --use_frozen_embeddings_as_input False --num_train_epochs 24 --max_eval_samples 500 --eval_steps 25000 --warmup_steps 100000 --bf16=1 --use_wandb=1"
    trainer = analyze_utils.load_trainer(checkpoint_folder, args_str)
    metrics = trainer.evaluate()
    # {'eval_loss': 1.0522613525390625,
    # 'eval_bleu_score': 31.552541624779003,
    # 'eval_accuracy': 0.7386067708333334,
    # 'eval_perplexity': 2.8641205867926014,
    # 'eval_runtime': 4.3916,
    # 'eval_samples_per_second': 113.855,
    # 'eval_steps_per_second': 0.911}
    assert pytest.approx(metrics["eval_bleu_score"]) == 31.552541624779003
