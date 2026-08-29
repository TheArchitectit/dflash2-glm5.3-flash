#!/usr/bin/env python3
"""Convert dump_target_hiddens output (HID1 binary) to hiddens.npz.

Layout: u32 magic, i32 n_tokens, i32 n_embd, u32 n_layers,
        i32 tokens[n_tokens],
        f32 layers[n_layers][n_tokens * n_embd]  (in TARGET_LAYERS order)
"""

import sys

import numpy as np

TARGET_LAYERS = [6, 15, 25, 34, 43]  # 1-indexed


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <hiddens.bin> <hiddens.npz>")
        return 1
    with open(sys.argv[1], "rb") as f:
        magic, = np.frombuffer(f.read(4), dtype=np.uint32)
        assert magic == 0x48494431, hex(magic)
        n_tokens, n_embd = np.frombuffer(f.read(8), dtype=np.int32)
        n_layers, = np.frombuffer(f.read(4), dtype=np.uint32)
        tokens = np.frombuffer(f.read(4 * n_tokens), dtype=np.int32)
        arrays = {"tokens": tokens}
        for k, il in enumerate(TARGET_LAYERS[:n_layers]):
            a = np.frombuffer(f.read(4 * n_tokens * n_embd), dtype=np.float32)
            arrays[f"layer_{il}"] = a.reshape(n_tokens, n_embd).copy()
    np.savez(sys.argv[2], **arrays)
    print(f"wrote {sys.argv[2]}: {n_tokens} tokens, {n_embd} dims, {n_layers} layers")
    for k in arrays:
        print(f"  {k}: {arrays[k].shape} mean|.|={np.abs(arrays[k]).mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
