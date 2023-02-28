from typing import Dict, List, Optional, Tuple

import math
import random

import evaluate
import torch
import tqdm
import transformers


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


class InversionTrainer(transformers.Trainer):
    def __init__(
            self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ###################################################### 
        self.metric_accuracy = evaluate.load("accuracy")
        self.metric_bleu = evaluate.load("sacrebleu")
        self.compute_metrics = self.compute_metrics_func
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
    
    def _log_preds_table(self, table_key: str, decoded_preds: List[str], decoded_labels: List[str]):
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
        wandb.log({table_key: table})
    
    def _get_eval_preds(self, n: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Iterates through eval dataset and does decoding.

        TODO: do this better. We shouldn't need to iterate through eval set twice
        but I don't want to copy 1000 lines of code to change their eval loop...

        Probably want custom eval eventually. Also this depends on eval data being
        in the same order which is annoying.
        """
        assert not self.model.training
        eval_dataloader = self.get_eval_dataloader()

        all_preds = []
        all_labels = []
        for step, inputs in enumerate(tqdm.tqdm(eval_dataloader, desc='generating from val', leave=False)):
            # https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
            inputs_cuda = {k: v.to(self.args.device) for k,v in inputs.items()}
            gen_kwargs = {
                'max_length': inputs['input_ids'].shape[1],
                'num_beams': 1,
                'do_sample': False,
            }
            with torch.no_grad():
                generated_text = self.model.generate(
                    inputs=inputs_cuda,
                    generation_kwargs=gen_kwargs
                )
            all_preds.extend(generated_text.cpu().tolist())
            all_labels.extend(inputs['input_ids'].cpu().tolist())
            if len(all_preds) >= n: break
        
        return all_preds, all_labels

    
    def _get_train_preds(self, n: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Iterates through eval dataset and does decoding.

        TODO: do this better. We shouldn't need to iterate through eval set twice
        but I don't want to copy 1000 lines of code to change their eval loop...

        Probably want custom eval eventually. Also this depends on eval data being
        in the same order which is annoying.
        """
        assert not self.model.training
        train_dataloader = self.get_train_dataloader()

        all_preds = []
        all_labels = []
        for step, inputs in enumerate(tqdm.tqdm(train_dataloader, desc='generating from train', leave=False)):
            # https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
            inputs_cuda = {k: v.to(self.args.device) for k,v in inputs.items()}
            gen_kwargs = {
                'max_length': inputs['input_ids'].shape[1],
                'num_beams': 1,
                'do_sample': False,
            }
            with torch.no_grad():
                generated_text = self.model.generate(
                    inputs=inputs_cuda,
                    generation_kwargs=gen_kwargs
                )
            all_preds.extend(generated_text.cpu().tolist())
            all_labels.extend(inputs['input_ids'].cpu().tolist())

            if len(all_preds) >= n:
                break
        
        return all_preds, all_labels
    
    
    def compute_metrics_func(self, eval_preds):
        preds  = eval_preds.predictions
        labels = eval_preds.label_ids

        assert len(labels), "got empty labels for eval"
        
        assert torch.tensor(preds).shape == torch.tensor(labels).shape
        
        # Get decoded text (note that this is *different*) than 'preds', which
        # is used to compute the loss.
        preds_sample, preds_sample_labels = self._get_eval_preds(n=1000)

        # Log BLEU, log table of text.
        decoded_preds = self.tokenizer.batch_decode(preds_sample, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(preds_sample_labels, skip_special_tokens=True)
        raw_bleu_result = self.metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_result = { "bleu_score": raw_bleu_result["score"]}
        self._log_preds_table(table_key="val_text_preds", decoded_preds=decoded_preds, decoded_labels=decoded_labels)

        # Log table for train data.
        train_preds_sample, train_preds_sample_labels = self._get_train_preds(n=50)
        decoded_train_preds = self.tokenizer.batch_decode(train_preds_sample, skip_special_tokens=True)
        decoded_train_labels = self.tokenizer.batch_decode(train_preds_sample_labels, skip_special_tokens=True)
        self._log_preds_table(table_key="train_text_preds", decoded_preds=decoded_train_preds, decoded_labels=decoded_train_labels)
        train_raw_bleu_result = self.metric_bleu.compute(predictions=decoded_train_preds, references=decoded_train_labels)
        # train_bleu_result = { "bleu_score": train_raw_bleu_result["score"]}

        # preds have the same shape as the labels.
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        accuracy_result = self.metric_accuracy.compute(predictions=preds, references=labels)

        return { **bleu_result, **accuracy_result }

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
        outputs = model(inputs=inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
    
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    

        