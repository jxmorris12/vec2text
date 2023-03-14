""" Reads embeddings from a file, converts to binary and writes binary strings to a text file in a folder.

Written: 2023-03-06
"""

import argparse
import os
import pickle
import struct

import numpy as np
import tqdm


num_workers = len(os.sched_getaffinity(0))


def parse_args() -> argparse.ArgumentParser:
        # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description='Get embeddings from a pre-trained model')
    parser.add_argument('--embeddings_file', '--e',
        type=str,
        required=True,
        help='The file to load embeddings from',
    )
    args = parser.parse_args()
    assert os.path.exists(args.embeddings_file), f"file not found {args.embeddings_file}"
    return args

def binary_str(num: float) -> str:
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def binary(num: float) -> np.ndarray:
    # adapted from https://stackoverflow.com/a/16444778
    s = binary_str(num)
    return np.array([int(c) for c in s])

def emb_to_binary(emb: np.ndarray) -> np.ndarray:
    print('embeddings shape:', emb.shape)
    out_arr = np.zeros((emb.shape[0], emb.shape[1], 32))
    out_arr.shape
    for i in tqdm.trange(emb.shape[0]):
        for j in range(emb.shape[1]):
                out_arr[i, j] = binary(emb[i, j].item())
    return out_arr

def main():
    args = parse_args()

    emb = pickle.load(open(args.embeddings_file, 'rb'))

    b = emb_to_binary(emb=emb)
    assert b.shape == (*emb.shape, 32)

    out_folder = args.embeddings_file.rstrip('.p')
    out_file_full = open(args.embeddings_file.rstrip('.p') + '_full.txt', 'w')
    print(f'writing to folder {out_folder}')
    os.makedirs(out_folder, exist_ok=False)

    for i in tqdm.trange(len(b), desc='writing embeddings to disk'):
        emb_chars = map(str, b[i].flatten().astype(int).tolist())
        emb_str = ''.join(emb_chars)
        open(os.path.join(out_folder, f'{i}.txt'), 'w').write(emb_str)
        out_file_full.write(emb_str + '\n')

        if i == 0:
            tqdm.tqdm.write(f'sample embedding (256 chars): {emb_str[:256]}')
    out_file_full.close()
    
    print(f'wrote {len(b)} binary embeddings to {out_folder}. :)')
    
    
if __name__ == '__main__': main()