#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
生成一个极小的随机 .clb 模型，用于端到端验证 Colibri 推理链路。
不依赖 numpy / transformers，只用 Python 标准库。

用法:
    python3 tools/make_tiny_model.py --output ./models/tiny_random.clb
"""
import argparse
import random
import struct
from pathlib import Path

HDR_SIZE = 128
MAGIC = b"COLIBRI\x00"
VERSION = 1
DTYPE_F32 = 1


def f32(x):
    return struct.pack("<f", x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # ---------- 超参数 ----------
    n_vocab    = 256
    n_embd     = 32
    n_layer    = 1
    n_head     = 4
    n_head_kv  = 4
    n_ff       = 64
    n_ctx      = 128
    rope_theta = 10000.0
    rms_eps    = 1e-5
    head_dim   = n_embd // n_head

    # ---------- Tokenizer 区 ----------
    # id 0 = <s> (BOS), id 1 = </s> (EOS), id 2..257 = 单字节 0..255
    tokens = [b"<s>", b"</s>"] + [bytes([b]) for b in range(256)]
    n_tokens = len(tokens)

    tok_blob = bytearray()
    tok_blob += struct.pack("<I", 0)                    # tok_model_len (预留)
    tok_blob += struct.pack("<I", n_tokens)
    for t in tokens:
        tok_blob += struct.pack("<I", len(t)) + t
    tok_blob += struct.pack("<I", 0)                    # n_merges = 0

    # ---------- 张量表 ----------
    tensors = []

    def add(name, *dims):
        ne = list(dims) + [1] * (4 - len(dims))
        tensors.append((name, ne))

    add("token_embd.weight",   n_embd, n_vocab)
    add("output_norm.weight",  n_embd)
    add("output.weight",       n_embd, n_vocab)

    for i in range(n_layer):
        add(f"blk.{i}.attn_norm.weight",   n_embd)
        add(f"blk.{i}.attn_q.weight",      n_embd, n_head * head_dim)
        add(f"blk.{i}.attn_k.weight",      n_embd, n_head_kv * head_dim)
        add(f"blk.{i}.attn_v.weight",      n_embd, n_head_kv * head_dim)
        add(f"blk.{i}.attn_output.weight", n_head * head_dim, n_embd)
        add(f"blk.{i}.ffn_norm.weight",    n_embd)
        add(f"blk.{i}.ffn_gate.weight",    n_embd, n_ff)
        add(f"blk.{i}.ffn_up.weight",      n_embd, n_ff)
        add(f"blk.{i}.ffn_down.weight",    n_ff, n_embd)

    table = bytearray()
    data  = bytearray()

    for name, ne in tensors:
        nb     = name.encode("utf-8")
        offset = len(data)

        table += struct.pack("<I", len(nb)) + nb
        table += struct.pack("<BBH", DTYPE_F32, 4, 0)
        table += struct.pack("<Q", offset)
        table += struct.pack("<4q", *ne)

        total = ne[0] * ne[1] * ne[2] * ne[3]

        # norm 权重接近 1.0，其他接近 0
        if "norm" in name:
            vals = [1.0 + random.uniform(-0.01, 0.01) for _ in range(total)]
        else:
            vals = [random.uniform(-0.1, 0.1) for _ in range(total)]

        data += b"".join(f32(v) for v in vals)

    # ---------- Header ----------
    hdr = bytearray(HDR_SIZE)
    hdr[0:8]   = MAGIC
    hdr[8:12]  = struct.pack("<I", VERSION)
    hdr[12:16] = struct.pack("<I", 0)
    hdr[16:20] = struct.pack("<I", n_vocab)
    hdr[20:24] = struct.pack("<I", n_embd)
    hdr[24:28] = struct.pack("<I", n_layer)
    hdr[28:32] = struct.pack("<I", n_head)
    hdr[32:36] = struct.pack("<I", n_head_kv)
    hdr[36:40] = struct.pack("<I", n_ff)
    hdr[40:44] = struct.pack("<I", n_ctx)
    hdr[44:48] = struct.pack("<I", len(tensors))
    hdr[48:52] = struct.pack("<f", rope_theta)
    hdr[52:56] = struct.pack("<f", rms_eps)
    hdr[56:60] = struct.pack("<I", 0)

    # ---------- 组装 ----------
    body = bytes(hdr) + bytes(tok_blob) + bytes(table)
    pad  = (-len(body)) % 64
    body += b"\x00" * pad

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(body)
        f.write(bytes(data))

    print(f"wrote {out}  ({out.stat().st_size} bytes, {len(tensors)} tensors)")
    print(f"  vocab={n_vocab} embd={n_embd} layer={n_layer} "
          f"head={n_head}/{n_head_kv} ff={n_ff}")


if __name__ == "__main__":
    main()