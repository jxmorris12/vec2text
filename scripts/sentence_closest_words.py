import sys
sys.path.append('/home/jxm3/research/retrieval/inversion')

import torch
import tqdm
import transformers

from models import (
    load_embedder_and_tokenizer,
    load_encoder_decoder,
    InversionModel
)
from utils import embed_all_tokens

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    embedder, embedder_tokenizer = (
        load_embedder_and_tokenizer(name="dpr")
    )
    model = InversionModel(
        embedder=embedder,
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
        
        a = embedder_tokenizer([a], return_tensors='pt').to(device)
        emb_a = model.call_embedding_model(**a)

        sims = torch.nn.CosineSimilarity(dim=1)(emb_a, word_embeddings)
        topk_sims = sims.topk(10, dim=0)
        for token_id, value in zip(topk_sims.indices, topk_sims.values):
            token = embedder_tokenizer.decode([token_id])
            print(f'\t{token}\t{value:.3f}')

        print()

    print("goodbye :)")

    
if __name__ == '__main__': main()