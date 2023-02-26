from typing import Dict, List, Optional

import math
import random

import evaluate
import torch
import transformers


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


class InversionTrainer(transformers.Trainer):
    embedder: torch.nn.Module # model to get embeddings from.

    def __init__(
            self, *args, embedder: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        ######################################################
        self.embedder = embedder.to(device)
        ######################################################
        embedder_dim = self.embedder.config.hidden_size
        encoder_hidden_dim = self.model.config.hidden_size
        self.embedding_transform = torch.nn.Linear(
            embedder_dim, encoder_hidden_dim
        ).to(device)
        ###################################################### 
        self.metric_accuracy = evaluate.load("accuracy")
        self.metric_bleu = evaluate.load("sacrebleu")
        self.compute_metrics = self.compute_metrics_func
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
    
    def _log_preds_table(self, decoded_preds: List[str], decoded_labels: List[str]):
        if not self.args.use_wandb:
            return

        num_rows = 50
        idxs = random.choices(range(len(decoded_preds)), k=min(len(decoded_preds), num_rows))

        data = []
        for idx in idxs:
            data.append([decoded_labels[idx], decoded_preds[idx]])
        
        import wandb
        table = wandb.Table(
            columns=["Original", "Decoded"],
            data=data
        )
        wandb.log({"val_text_preds": table})
    
    def compute_metrics_func(self, eval_preds):
        preds, labels = eval_preds
        
        # Get decoded preds and BLEU, log table of text.
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        raw_bleu_result = self.metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_result = { "bleu_score": raw_bleu_result["score" ]}
        self._log_preds_table(decoded_preds=decoded_preds, decoded_labels=decoded_labels)

        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        accuracy_result = self.metric_accuracy.compute(predictions=preds, references=labels)

        return { **bleu_result, **accuracy_result }
    
    def call_both_models(
            self,
            model: transformers.AutoModelForCausalLM,
            inputs: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            assert not self.embedder.training
            embeddings = self.embedder(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            ).pooler_output
        
        # TODO: implement mean-pooling so we can test BERT sentence
        # embeddings vs SimCSE vs Contriever etc fairly.

        embeddings = self.embedding_transform(embeddings)
        inputs_embeds = embeddings.unsqueeze(dim=1).repeat((1, 32, 1)) # TODO make this fancier/abstracted.
        return model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones((inputs_embeds.shape[0], 32), device=inputs_embeds.device),
            # 
            decoder_input_ids=inputs["input_ids"],
            decoder_attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )

    def evaluation_loop(self, *args, **kwargs) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics.
        
        Override to compute ppl from eval loss.
        """
        output = super().evaluation_loop(*args, **kwargs)

        try:
            perplexity = math.exp(output.metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        output.metrics["eval_perplexity"] = perplexity
    
        return output

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. 
        
        We override this to change call to `model()`  to `self.call_both_models()`.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = self.call_both_models(model, inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    

        