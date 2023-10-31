from typing import Dict, Optional

import torch

from vec2text.trainers.inversion import InversionTrainer


class InversionFromLogitsTrainer(InversionTrainer):
    """Custom trainer for inverting from logits. Contains special
    decoding methods that we can only use here, mostly that
    have to do with conditioning on a suffix.
    """

    generation_method: Optional[str] = None
    test_suffixes = [
        "The purple elephant danced gracefully in the moonlight.",
        "Tacos and ice cream are my favorite combination for dinner.",
        "In a parallel universe, cats rule the world with laser pointers.",
        "The sunflower swayed in the gentle breeze, singing a lullaby.",
        "Quantum mechanics explains the behavior of particles at the smallest scales.",
        "Lost in a labyrinth of thoughts, she found a path to inner peace.",
        "Chocolate chip cookies are the key to happiness, or so they say.",
        "The detective followed the trail of breadcrumbs to solve the mystery.",
        "Deep in the enchanted forest, fairies whispered secrets to the trees.",
        "The robot chef prepared a gourmet meal with precision and flair.",
    ]  # from chatGPT

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        if self.generation_method == "suffix_ensemble":
            return self.generate_suffix_ensemble(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
            )
        elif self.generation_method == "length_check":
            return self.generate_and_check_length(
                inputs=inputs, generation_kwargs=generation_kwargs
            )
        else:
            return self.model.generate(
                inputs=inputs, generation_kwargs=generation_kwargs
            )

    def generate_suffix_ensemble(
        self, inputs: Dict, generation_kwargs: Dict
    ) -> torch.Tensor:
        batch_size = len(inputs["embedder_input_ids"])
        ex = self.embedder_tokenizer.batch_decode(
            inputs["embedder_input_ids"], skip_special_tokens=True
        )
        suffixes = self.test_suffixes
        num_suffixes = len(suffixes)
        ex_with_suffix_str = [(e + s) for e in ex for s in suffixes]
        ex_with_suffix_tokenized = self.embedder_tokenizer(
            ex_with_suffix_str,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.args.device)

        ex_with_suffix_tokenized_inverter = self.tokenizer(
            ex_with_suffix_str,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.args.device)

        prefix_lengths = self.tokenizer(
            ex, padding=True, truncation=False, return_tensors="pt"
        ).attention_mask.sum(dim=1)
        print("prefix_lengths:", prefix_lengths)

        # TODO: figure out how to do this with tensors
        suffix_ids = []
        max_length = -1
        for j in range(num_suffixes):
            for i, p in enumerate(prefix_lengths):
                suffix_ids_i = ex_with_suffix_tokenized_inverter.input_ids[i + j, p:]

                # fix padding for something I did by accident during training.
                # an artifact if you will.
                suffix_ids_i = [
                    token_id
                    for token_id in suffix_ids_i.tolist()
                    if token_id not in [0, 1, 2]
                ]

                # put BOS token at the end to match training labels.
                suffix_ids_i.append(1)

                max_length = max(max_length, len(suffix_ids_i))
                suffix_ids.append(suffix_ids_i)

        # pad.
        for i in range(len(suffix_ids)):
            num_pad_tokens = max_length - len(suffix_ids[i])
            suffix_ids[i].extend([0] * num_pad_tokens)

        suffix_ids = torch.tensor(suffix_ids).to(self.args.device)
        assert suffix_ids.shape[0] == (num_suffixes * batch_size)

        decoder_start_token_id = (
            self.model.encoder_decoder.config.decoder_start_token_id
        )
        decoder_input_ids = (
            torch.ones(
                (batch_size * num_suffixes, 1),
                dtype=torch.long,
                device=self.model.device,
            )
            * decoder_start_token_id
        )
        with torch.no_grad():
            frozen_embeddings = self.model.call_embedding_model(
                **ex_with_suffix_tokenized
            )

        eos_token_id = self.model.encoder_decoder.config.eos_token_id
        # past_key_values = None
        for step in range(2, 63):
            with torch.no_grad():
                output = self.model(
                    # embedder_input_ids=ex_with_suffix_tokenized.input_ids,
                    # embedder_attention_mask=ex_with_suffix_tokenized.attention_mask,
                    embedder_input_ids=None,
                    embedder_attention_mask=None,
                    suffix_ids=suffix_ids,
                    # suffix_ids=None,
                    decoder_input_ids=decoder_input_ids,
                    # past_key_values=past_key_values,
                    frozen_embeddings=frozen_embeddings,
                    use_cache=True,
                )
            # past_key_values = output.past_key_values

            logits = output.logits[:, -1, :]
            all_next_tokens = logits.argmax(dim=1).reshape((batch_size, num_suffixes))

            main_next_tokens = all_next_tokens[:, 0]
            other_next_tokens = all_next_tokens[:, 1:]

            all_others_agree = (
                other_next_tokens[:, :1] == other_next_tokens[:, 1:]
            ).all(dim=-1)

            disagreement_from_main = other_next_tokens[:, 0] != main_next_tokens

            if (all_others_agree & disagreement_from_main).any():
                import pdb

                pdb.set_trace()

            next_tokens = torch.where(
                all_others_agree, other_next_tokens[:, 0], main_next_tokens
            )
            next_tokens = next_tokens[:, None]

            # # ensembling
            # logits = output.logits[:, -1, :]
            # logits = (
            #     logits.reshape((batch_size, num_suffixes, -1))
            #     .mean(dim=1)
            #     .log_softmax(dim=1)
            # )

            # next_tokens = (
            #     logits.argmax(dim=1, keepdim=True)
            # )
            next_tokens = next_tokens.repeat((1, num_suffixes)).reshape(
                (batch_size * num_suffixes, 1)
            )

            # greedy sampling
            decoder_input_ids = torch.cat((decoder_input_ids, next_tokens), dim=1)

            # break early if we got EOS for everything in the batch
            all_eos_seen = (decoder_input_ids == eos_token_id).any(dim=1).all()
            if all_eos_seen:
                break

        decoder_input_ids = decoder_input_ids.reshape((batch_size, num_suffixes, -1))
        decoder_input_ids = decoder_input_ids[:, 0, :]
        eos_start_idx = (decoder_input_ids == eos_token_id).long().argmax(dim=1)

        # this fancy bit of code sets everything after eos token to padding.
        seq_length = decoder_input_ids.shape[1]
        eos_start_idx = eos_start_idx.where(eos_start_idx > 0, seq_length)
        decoder_input_ids = decoder_input_ids.where(
            torch.arange(seq_length, device=self.model.device)[None]
            <= eos_start_idx[:, None],
            self.model.encoder_decoder.config.pad_token_id,
        )
        return decoder_input_ids

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
