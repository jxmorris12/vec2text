import pytest

from vec2text import analyze_utils


def test_load_model_seqlen32():
    _experiment_, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
        "jxm/gtr__nq__32__correct"
    )
    metrics = trainer.evaluate(eval_dataset=trainer.eval_dataset["nq"])
    # {'eval_loss': 1.0522613525390625,
    # 'eval_bleu_score': 31.552541624779003,
    # 'eval_accuracy': 0.7386067708333334,
    # 'eval_perplexity': 2.8641205867926014,
    # 'eval_runtime': 4.3916,
    # 'eval_samples_per_second': 113.855,
    # 'eval_steps_per_second': 0.911}
    assert pytest.approx(metrics["eval_bleu_score"]) == 49.79286134672497
