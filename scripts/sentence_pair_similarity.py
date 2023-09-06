"""Embeds two sentences and computes their similarity. Outputs to command line.

Written: 2023-03-02
"""
import sys

sys.path.append("/home/jxm3/research/retrieval/inversion")

import torch
from models import InversionModel, load_embedder_and_tokenizer, load_encoder_decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    embedder, embedder_tokenizer = load_embedder_and_tokenizer(name="dpr")
    model = InversionModel(
        embedder=embedder,
        embedder_tokenizer=embedder_tokenizer,
        tokenizer=tokenizer,
        encoder_decoder=load_encoder_decoder(
            model_name="t5-small",
        ),
        num_repeat_tokens=6,
        embedder_no_grad=True,
        freeze_strategy="none",
    )
    model.to(device)
    while True:
        a = input("Enter sentence A (q to quit): ")
        if a == "q":
            break
        b = input("Enter sentence B: ")

        a = embedder_tokenizer([a], return_tensors="pt").to(device)
        b = embedder_tokenizer([b], return_tensors="pt").to(device)
        emb_a = model.call_embedding_model(**a)
        emb_b = model.call_embedding_model(**b)

        similarity = torch.nn.CosineSimilarity(dim=1)(emb_a, emb_b).item()

        print(f"Similarity = {similarity:.4f}")
        print()

    print("goodbye :)")


if __name__ == "__main__":
    main()
