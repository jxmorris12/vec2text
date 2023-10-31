import sys

sys.path.append("/home/jxm3/research/retrieval/inversion")

import argparse
import collections
import math
import os
import struct

import numpy as np
import pandas as pd
import torch
import tqdm
import transformers
from data_helpers import NQ_DEV, load_dpr_corpus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def entropy__bits(p: float) -> float:
    # entropy
    eps = 1e-16
    return -1 * (p * math.log2(p + eps) + (1 - p) * math.log2(1 - p + eps))


def binary_str(num: float) -> str:
    return "".join("{:0>8b}".format(c) for c in struct.pack("!f", num))


def binary(num: float) -> np.ndarray:
    # adapted from https://stackoverflow.com/a/16444778
    s = binary_str(num)
    return np.array([int(c) for c in s])


def emb_to_binary(emb: np.ndarray) -> np.ndarray:
    out_arr = np.zeros((emb.shape[0], emb.shape[1], emb.shape[2], 32))
    out_arr.shape
    for i in tqdm.trange(emb.shape[0]):
        for j in range(emb.shape[1]):
            for k in range(emb.shape[2]):
                out_arr[i, j, k] = binary(emb[i, j, k].item())
    return out_arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get embeddings from a pre-trained model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the pre-trained model to get embeddings from",
    )
    parser.add_argument(
        "--randomize_weights",
        default=False,
        action="store_true",
        help="Initialize weights randomly instead of from a pretrained checkpoint",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Max number of examples to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Inference batch size",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=32,
        help="Inference batch size",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    #
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = f"{args.model_name}__{args.max_seq_length}__{args.randomize_weights}__{args.n}.p"
    out_path = os.path.join(dir_path, os.pardir, "embeddings", "binary", filename)
    out_path = os.path.normpath(out_path)
    print(f"writing embeddings to {out_path}")
    #
    vd = load_dpr_corpus(NQ_DEV)
    #
    batch_size = args.batch_size
    msl = args.max_seq_length
    n = args.n
    #
    mn = args.model_name
    if args.randomize_weights:
        print(f"Loading model {mn} with randomly initialized weights.")
        # pass the config to model constructor instead of from_pretrained
        # this creates the model as per the params in config
        # but with weights randomly initialized
        config = transformers.AutoConfig.from_pretrained(mn)
        m = transformers.AutoModel.from_config(config)
        t = transformers.AutoTokenizer.from_pretrained(mn)
    else:
        m = transformers.AutoModel.from_pretrained(mn)
    #
    t = transformers.AutoTokenizer.from_pretrained(mn)
    m.to(device)
    #
    i = 0
    pbar = tqdm.tqdm(total=n, leave=False)
    all_hidden_states = collections.defaultdict(list)
    while i < n:
        #
        # tokenize text
        text = vd[i : i + batch_size]["text"]
        tt = t(
            text, truncation=True, padding=True, max_length=msl, return_tensors="pt"
        ).to(device)
        #
        # feed to model
        with torch.no_grad():
            e = m.embeddings(input_ids=tt["input_ids"])
            o = m(**tt, output_hidden_states=True)
        # add embeddings as layer 0
        all_hidden_states[0].append(e.cpu())
        #
        # aggregate hidden states
        # o.hidden_states is a tuple of length (n_layers).
        # each entry has shape (b, s, d).
        hs = o.hidden_states
        for j in range(len(hs)):
            all_hidden_states[j + 1].append(hs[j].cpu())

        i += batch_size
        pbar.update(batch_size)

    all_hidden_states = {
        k: torch.cat(v, dim=0).numpy() for k, v in all_hidden_states.items()
    }

    # add mean pooling as "last layer"
    all_hidden_states[len(all_hidden_states)] = all_hidden_states[
        len(all_hidden_states) - 1
    ].mean(axis=1, keepdims=True)

    # convert all floats to binary :-)
    binary_hidden_states = {k: emb_to_binary(v) for k, v in all_hidden_states.items()}

    # average over dataset to produce probability heatmap
    hidden_states_heatmaps = {
        k: v.mean(axis=0) for k, v in binary_hidden_states.items()
    }

    # compute bits from probabilities
    hidden_states_bits = {
        k: np.vectorize(entropy__bits)(v) for k, v in hidden_states_heatmaps.items()
    }

    data = []

    for layer_idx, layer_data in hidden_states_bits.items():
        print(
            "Unrolling bits from layer", layer_idx
        )  # (TODO optimize - looping by hand like this actually takes a few seconds..)
        for seq in range(layer_data.shape[0]):
            for emb_dim in range(layer_data.shape[1]):
                for bit in range(layer_data.shape[2]):
                    bit_value = layer_data[seq, emb_dim, bit]
                    data.append([layer_idx, seq, emb_dim, bit, bit_value])

    df = pd.DataFrame(
        data, columns=["layer", "sequence", "embedding_dim", "bit_idx", "bit_value"]
    )
    # for n,v in vars(args).items():
    #     df[n] = v
    df.to_pickle(open(out_path, "wb"))
    print(f"wrote {len(df)} embeddings to {out_path}")


if __name__ == "__main__":
    main()
