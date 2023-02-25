from typing import Dict, Optional

import torch
import transformers


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InversionTrainer(transformers.Trainer):
    embedder: torch.nn.Module # model to get embeddings from.

    def __init__(
            self, *args, embedder: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        # 
        self.embedder = embedder.to(device)
        # 
        embedder_dim = self.embedder.config.hidden_size
        encoder_hidden_dim = self.model.config.hidden_size
        self.embedding_transform = torch.nn.Linear(
            embedder_dim, encoder_hidden_dim
        ).to(device)
        # 
    
    def call_both_models(
            self,
            model: transformers.AutoModelForCausalLM,
            inputs: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            embeddings = self.embedder(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            ).pooler_output
        
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
    

        