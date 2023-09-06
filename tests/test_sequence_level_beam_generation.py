from vec2text import aliases


def test_generation():
    # inv_trainer = aliases.load_trainer_from_alias("openai_msmarco__msl128__100epoch")
    corr_experiment, corr_trainer = aliases.load_experiment_and_trainer_from_alias(
        "gtr_nq__msl32_beta__correct"
    )
    # inv_trainer = corr_trainer.inversion_trainer
    # corr_trainer.precompute_hypotheses()
    corr_trainer.model.eval()
    print()

    corr_trainer.args.per_device_eval_batch_size = 100
    corr_trainer.gen_kwargs = {
        "early_stopping": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "do_sample": False,
        "no_repeat_ngram_size": 0,
        "min_length": 32,
        "max_length": 32,
    }

    eval_batch = next(
        iter(
            corr_trainer.get_eval_dataloader(
                eval_dataset=corr_trainer.eval_dataset["nq"]
            )
        )
    )
    one_eval_batch = {k: v[32:34] for k, v in eval_batch.items()}
    one_eval_batch = {
        k: v.to(corr_trainer.args.device) for k, v in one_eval_batch.items()
    }

    print(
        corr_trainer.embedder_tokenizer.batch_decode(
            one_eval_batch["embedder_input_ids"], skip_special_tokens=True
        )
    )

    gen_ids = corr_trainer.generate(
        inputs=one_eval_batch,
        generation_kwargs=corr_trainer.gen_kwargs,
        num_recursive_steps=10,
        sequence_beam_width=10,
    )
    print(
        corr_trainer.embedder_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    )
