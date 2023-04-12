import torch
import tqdm
import transformers

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
    model.embedder.eval()
    batch_size = 1024
    all_token_embeddings = []
    V = tokenizer.vocab_size
    #
    # DPR has CLS and SEP.
    # GTR has no CLS or start token at all, and has EOS at the end.
    CLS = tokenizer.cls_token_id
    SEP = (tokenizer.sep_token_id) or (tokenizer.eos_token_id)
    assert SEP is not None
    #
    device = next(model.parameters()).device
    pbar = tqdm.tqdm(desc='generating token embeddings', colour='#008080', total=V, leave=False)
    while i < V:
        # 
        minibatch_size = min(V-i, batch_size)
        inputs = torch.arange(i, min(i+minibatch_size, V))
        # 
        if (CLS is not None):
            input_ids = torch.stack([torch.tensor([CLS]).repeat(len(inputs)), inputs, torch.tensor([SEP]).repeat(len(inputs))]).T
        else:
            input_ids = torch.stack([inputs, torch.tensor([SEP]).repeat(len(inputs))]).T
        input_ids = input_ids.to(device)
        # 
        attention_mask = torch.ones_like(input_ids, device=device)
        # 
        with torch.no_grad():
            token_embeddings = emb(model, input_ids, attention_mask)        
        all_token_embeddings.extend(token_embeddings)
        i += batch_size
        pbar.update(batch_size)
    # 
    all_token_embeddings = torch.stack(all_token_embeddings)
    print('all_token_embeddings.shape:', all_token_embeddings.shape)
    assert all_token_embeddings.shape == (tokenizer.vocab_size, 768)

    all_token_embeddings /= all_token_embeddings.norm(p=2, dim=1, keepdim=True)
    return all_token_embeddings
