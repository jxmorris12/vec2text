"""Embeds a sentence and outputs the closest tokens as measured by emb(token).

Written: 2023-03-05
"""
import sys

sys.path.append("/home/jxm3/research/retrieval/inversion")

import torch
import tqdm
import transformers
from models import InversionModel, load_embedder_and_tokenizer, load_encoder_decoder
from utils import embed_all_tokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# embedder_model_name = "dpr"
embedder_model_name = "gtr_base"


def main():
    embedder, embedder_tokenizer = load_embedder_and_tokenizer(name=embedder_model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
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
    word_embeddings = embed_all_tokens(model, embedder_tokenizer)

    while True:
        a = input("Enter sentence (q to quit): ")
        if a == "q":
            break

        a = embedder_tokenizer([a], return_tensors="pt").to(device)
        emb_a = model.call_embedding_model(**a)

        sims = torch.nn.CosineSimilarity(dim=1)(emb_a, word_embeddings)
        topk_sims = sims.topk(40, dim=0)
        for token_id, value in zip(topk_sims.indices, topk_sims.values):
            token = embedder_tokenizer.decode([token_id])
            print(f"\t{token}\t{value:.2f}")

        print()

    print("goodbye :)")


if __name__ == "__main__":
    main()
