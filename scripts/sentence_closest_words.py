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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def emb(
        model: torch.nn.Module, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
    with torch.no_grad():
        emb = model.call_embedding_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
    return emb


def embed_all_tokens(model: torch.nn.Module, tokenizer: transformers.AutoTokenizer):
    """Generates embeddings for all tokens in tokenizer vocab."""
    i = 0
    batch_size = 512
    all_token_embeddings = []
    V = tokenizer.vocab_size
    CLS = tokenizer.vocab['[CLS]']
    SEP = tokenizer.vocab['[SEP]']
    pbar = tqdm.tqdm(desc='generating token embeddings', colour='#008080', total=V)
    while i < V:
        # 
        minibatch_size = min(V-i, batch_size)
        inputs = torch.arange(i, i+minibatch_size)
        input_ids = torch.stack([torch.tensor([101]).repeat(len(inputs)), inputs, torch.tensor([102]).repeat(len(inputs))]).T
        input_ids = input_ids.to(device)
        # 
        attention_mask = torch.ones_like(input_ids, device=device)
        # 
        token_embeddings = emb(model, input_ids, attention_mask)
        all_token_embeddings.extend(token_embeddings)
        i += batch_size
        pbar.update(batch_size)
    # 
    all_token_embeddings = torch.stack(all_token_embeddings)
    print('all_token_embeddings.shape:', all_token_embeddings.shape)
    assert all_token_embeddings.shape == (30522, 768)
    return all_token_embeddings

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