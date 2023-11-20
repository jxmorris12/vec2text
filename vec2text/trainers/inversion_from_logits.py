from typing import Dict, Optional

import torch

from vec2text.trainers.inversion import InversionTrainer


class InversionFromLogitsTrainer(InversionTrainer):
    """Custom trainer for inverting from logits. Contains special
    decoding methods that we can only use here, mostly that
    have to do with conditioning on a suffix.
    """

    generation_method: Optional[str] = None

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        print("generate with method:", self.generation_method)
        if self.generation_method == "length_check":
            return self.generate_and_check_length(
                inputs=inputs, generation_kwargs=generation_kwargs
            )
        else:
            return self.model.generate(
                inputs=inputs, generation_kwargs=generation_kwargs
            )

    def generate_and_check_length(
        self, inputs: Dict, generation_kwargs: Dict
    ) -> torch.Tensor:
        with torch.no_grad():
            frozen_embeddings = self.model.call_embedding_model(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            )

        batch_size = len(inputs["embedder_input_ids"])

        closest_generations = None
        closest_generation_distances = None
        for length in range(1, 64):
            generation_kwargs["min_length"] = length
            generation_kwargs["max_length"] = length

            generations = self.model.generate(
                inputs=inputs, generation_kwargs=generation_kwargs
            )
            generations_str = self.tokenizer.batch_decode(
                generations, skip_special_tokens=True
            )
            generation_emb_tokenized = self.embedder_tokenizer(
                generations_str,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64,
            ).to(self.args.device)
            with torch.no_grad():
                new_embeddings = self.model.call_embedding_model(
                    **generation_emb_tokenized
                )
            new_distances = torch.nn.functional.kl_div(
                frozen_embeddings,
                new_embeddings,
                reduction="none",
                log_target=True,
            ).sum(dim=1)
            # new_distances = ((frozen_embeddings - new_embeddings) ** 2).sum(1)

            num_pad_tokens = 64 - generations.shape[1]
            pad_tokens = (
                torch.ones(
                    (batch_size, num_pad_tokens),
                    dtype=torch.long,
                    device=self.args.device,
                )
                * self.tokenizer.pad_token_id
            )
            generations = torch.cat((generations, pad_tokens), dim=1)
            if closest_generations is None:
                closest_generations = generations
                closest_generation_distances = new_distances
            else:
                closest_generations = torch.where(
                    (new_distances < closest_generation_distances)[:, None],
                    generations,
                    closest_generations,
                )
                closest_generation_distances = torch.where(
                    new_distances < closest_generation_distances,
                    new_distances,
                    closest_generation_distances,
                )

        return closest_generations
