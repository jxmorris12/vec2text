import transformers

import vec2text


def test_openai_corrector():
    inversion_model = vec2text.models.InversionModel.from_pretrained(
        "jxm/vec2text__openai_ada002__msmarco__msl128__hypothesizer"
    )
    model = vec2text.models.CorrectorEncoderModel.from_pretrained(
        "jxm/vec2text__openai_ada002__msmarco__msl128__corrector"
    )

    inversion_trainer = vec2text.trainers.InversionTrainer(
        model=inversion_model,
        train_dataset=None,
        eval_dataset=None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            inversion_model.tokenizer,
            label_pad_token_id=-100,
        ),
    )

    # backwards compatibility stuff
    model.config.dispatch_batches = None
    trainer = vec2text.trainers.Corrector(
        model=model,
        inversion_trainer=inversion_trainer,
        args=None,
        data_collator=vec2text.collator.DataCollatorForCorrection(
            tokenizer=inversion_trainer.model.tokenizer
        ),
    )
    trainer.sanity_decode()
