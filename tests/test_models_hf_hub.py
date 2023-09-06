import transformers
import vec2text


def test_openai_corrector():
    inversion_model = vec2text.models.InversionModel.from_pretrained("jxm/vec2text__openai_ada002__msmarco__msl128__hypothesizer")
    model = vec2text.models.CorrectorEncoderModel.from_pretrained("jxm/vec2text__openai_ada002__msmarco__msl128__corrector")

    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")

    inversion_trainer = vec2text.trainers.InversionTrainer(
        model=model,
        train_dataset=None,
        eval_dataset=None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        ),
    )

    trainer = vec2text.trainers.CorrectorTrainer(
        model=model,
        inversion_trainer=inversion_trainer,
        # args=self.training_args,
        data_collator=vec2text.collator.DataCollatorForCorrection(
            tokenizer=inversion_trainer.model.tokenizer
        ),
    )
    model.sanity_decode()

