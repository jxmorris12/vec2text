from typing import Any, Dict, List, Optional, Tuple, Union

import copy
import math
import random

import evaluate
import torch
import torch.nn as nn
import tqdm
import transformers

from models import PrefixReranker


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
        self.model.precompute_whitening_params(self.get_train_dataloader())
        self.gen_kwargs = {
            "early_stopping": False,
            # "no_repeat_ngram_size": 3,
            # From CtRL paper: We find that using a greedy sampling 
            # and θ ≈ 1.2 yields a good balance between truthful generation
            # and lack of repetition (arxiv.org/abs/1909.05858)
            "repetition_penalty": 1.2,
            "num_beams": 1,
            'do_sample': False,
        }
    
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

        gen_kwargs = copy.copy(self.gen_kwargs)

        all_preds = []
        all_labels = []
        for step, inputs in enumerate(tqdm.tqdm(eval_dataloader, desc='generating from val', leave=False)):
            # https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
            inputs_cuda = {k: v.to(self.args.device) for k,v in inputs.items()}
            gen_kwargs["min_length"] = gen_kwargs["max_length"] = inputs["input_ids"].shape[1]
            print("gen_kwargs:", gen_kwargs)
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

        gen_kwargs = copy.copy(self.gen_kwargs)

        all_preds = []
        all_labels = []
        for step, inputs in enumerate(tqdm.tqdm(train_dataloader, desc='generating from train', leave=False)):
            # https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
            inputs_cuda = {k: v.to(self.args.device) for k,v in inputs.items()}
            gen_kwargs["min_length"] = gen_kwargs["max_length"] = inputs["input_ids"].shape[1]
            with torch.no_grad():
                generated_text = self.model.generate(
                    inputs=inputs_cuda,
                    generation_kwargs=gen_kwargs,
                )
            all_preds.extend(generated_text.cpu().tolist())
            all_labels.extend(inputs['input_ids'].cpu().tolist())

            if len(all_preds) >= n:
                break
        
        return all_preds, all_labels
    
    
    def _compute_data_metrics(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        inputs_pad_tokens = (
            inputs["input_ids"] == self.model.tokenizer.pad_token_id
        ).sum(dim=1).float().mean().item()
        embedder_inputs_pad_tokens = (
            inputs["embedder_input_ids"] == self.model.embedder_tokenizer.pad_token_id
        ).sum(dim=1).float().mean().item()

        inputs_non_pad_tokens = inputs["input_ids"].shape[1] - inputs_pad_tokens
        embedder_inputs_non_pad_tokens = inputs["input_ids"].shape[1] - embedder_inputs_pad_tokens

        return {
            "encoder_decoder_inputs_pad_tokens": inputs_pad_tokens,
            "encoder_decoder_inputs_non_pad_tokens": inputs_non_pad_tokens,
            "embedder_inputs_pad_tokens": embedder_inputs_pad_tokens,
            "embedder_inputs_non_pad_tokens": embedder_inputs_non_pad_tokens,
        }
        
    
    def training_step(
            self,
            model: nn.Module,
            inputs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        """
        Performs a training step. we override to compute data-specific metrics.
        """
        metrics = self._compute_data_metrics(inputs=inputs)
        # self.log({ f"train/{k}": v for k,v in metrics.items() })
        return super().training_step(model, inputs)


    def compute_metrics_func(self, eval_preds):
        preds  = eval_preds.predictions
        labels = eval_preds.label_ids

        assert len(labels), "got empty labels for eval"
        
        assert torch.tensor(preds).shape == torch.tensor(labels).shape
        
        # Get decoded text –note that this is different than `preds`, which
        # is used to compute the loss.
        preds_sample, preds_sample_labels = self._get_eval_preds(n=1000)

        # Log BLEU, log table of text.
        decoded_preds = self.model.tokenizer.batch_decode(preds_sample, skip_special_tokens=True)
        decoded_labels = self.model.tokenizer.batch_decode(preds_sample_labels, skip_special_tokens=True)
        raw_bleu_result = self.metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_result = { "bleu_score": raw_bleu_result["score"]}
        self._log_preds_table(table_key="val_text_preds", decoded_preds=decoded_preds, decoded_labels=decoded_labels)

        print(decoded_preds[0])
        print(decoded_labels[0])
        print('\n\n')
        print(decoded_preds[1])
        print(decoded_labels[1])
        print('\n\n')
        print(decoded_preds[2])
        print(decoded_labels[2])

        # Compute sims of eval data using embedder.
        preds_sample = torch.tensor(preds_sample, device=self.args.device)[:128]
        preds_sample_labels = torch.tensor(preds_sample_labels, device=self.args.device)[:128]
        # Fix eos token on generated text.
        eos_token_id = self.model.embedder_tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_tokens = torch.ones((len(preds_sample), 1), dtype=preds_sample.dtype, device=self.args.device) * eos_token_id
            preds_sample = torch.cat((preds_sample[:, 1:], eos_tokens), dim=1)
        with torch.no_grad():
            preds_emb = self.model.call_embedding_model(
                input_ids=preds_sample,
                attention_mask=torch.ones_like(preds_sample, device=self.args.device)
            )
            labels_emb = self.model.call_embedding_model(
                input_ids=preds_sample_labels,
                attention_mask=torch.ones_like(preds_sample_labels, device=self.args.device)
            )
            emb_cos_sim = torch.nn.CosineSimilarity(dim=1)(preds_emb, labels_emb).mean().item()
            sim_result = {"emb_cos_sim": emb_cos_sim}

        # Log table for train data.
        train_preds_sample, train_preds_sample_labels = self._get_train_preds(n=50)
        decoded_train_preds = self.model.tokenizer.batch_decode(train_preds_sample, skip_special_tokens=True)
        decoded_train_labels = self.model.tokenizer.batch_decode(train_preds_sample_labels, skip_special_tokens=True)
        self._log_preds_table(table_key="train_text_preds", decoded_preds=decoded_train_preds, decoded_labels=decoded_train_labels)
        train_raw_bleu_result = self.metric_bleu.compute(predictions=decoded_train_preds, references=decoded_train_labels)
        # train_bleu_result = { "bleu_score": train_raw_bleu_result["score"]}

        # preds have the same shape as the labels.
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        accuracy_result = self.metric_accuracy.compute(predictions=preds, references=labels)

        return { **bleu_result, **accuracy_result, **sim_result }

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
    


class RerankingTrainer(transformers.Trainer):
    def __init__(
            self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO support gc?
        
    def _contrastive_loss(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        batch_size = len(e1)

        # Normalize vectors, so loss is cosine similarity (not raw
        # dot product!)
        e1 = e1 / e1.norm(p=2, dim=1, keepdim=True)
        e2 = e2 / e2.norm(p=2, dim=1, keepdim=True)

        similarity = (e1 @ e2.T)

        # This multiply-by-20 trick is used in BeIR and LaPRador:
        # "When calculating the loss, we apply a re-scaling trick
        # of multiplying the cosine similarity score by 20 for
        # better optimization (Thakur et al., 2021)."
        # 
        similarity *= 20  # TODO argparse: self.args.contrastive_temperature.exp()
        diagonal_idxs = torch.arange(batch_size, device=e1.device)
        loss = torch.nn.functional.cross_entropy(
            similarity, diagonal_idxs, 
            label_smoothing=0.0
        )
        if (loss.isnan()):
            raise RuntimeError("Loss is nan!")

        # outputs = {
        #     "query_embedding": e1.detach().cpu(),
        #     "document_embedding": e2.detach().cpu(),
        # }
        # accuracy = (similarity.argmax(dim=1) == diagonal_idxs).float().mean()
        # if self.args.use_wandb:
        #     # Log model-internal
        #     # todo: use self.log() instead?
        #     import wandb
        #     wandb.log({**metrics_q, **metrics_d, "accuracy": accuracy})
                
        return loss
    
    
    def generate_continuations(
        self,
        input_ids: torch.Tensor,
        continuation_length: int,
        ) -> torch.Tensor:
        raise NotImplementedError()

    def compute_loss(
            self,
            model: PrefixReranker,
            inputs: Dict[str, torch.Tensor],
            return_outputs: bool = False,
        ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        assert BIENCODER_DATA_KEYS <= set(inputs.keys()), f"invalid keys {inputs.keys()}"

        batch_size, seq_length = inputs['input_ids'].shape
        prefix_length = random.randint(1, seq_length)
        max_continuation_length = seq_length - prefix_length
        # The 10 here comes from rankgen (arxiv.org/pdf/2205.09726.pdf)
        # where they generate continuations of a random length
        # from 10 to msl.
        continuation_length = random.randint(10, max_continuation_length)

        prefixes = inputs['input_ids'][:, :prefix_length]
        true_continuations = inputs['input_ids'][:, :prefix_length+continuation_length]

        # TODO: Should we use beam search or nucleus sampling here?
        fake_continuations = self.generate_continuations(prefixes, continuation_length)

        true_prefix_embeds = self.model.embed_prefix(true_continuations)
        fake_prefix_embeds = self.model.embed_prefix(fake_continuations)
        # TODO is subsequent concat() correct?
        all_prefix_embeds = torch.stack((true_prefix_embeds, fake_prefix_embeds), dim=1)

        embedding_embeds = self.model.embedding_projection(inputs['frozen_embeddings'])

        return self._contrastive_loss(query_inputs, doc_inputs, no_sync_except_last=False)