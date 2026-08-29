#!/usr/bin/env python3
"""Sprint 3.3 — SGLang reference draft dump (golden test reference arm).

Faithful pure-torch re-implementation of the DFlash2 draft forward
(sglang/srt/models/dflash.py semantics, raw checkpoint weights, no SGLang
runtime objects), fed by the fixture hiddens from dump_target_hiddens (3.1).

Per dflash.py DFlashDecoderLayer.forward:
    h = input_layernorm(x [+ residual])
    h_conv, coeff = attention_conv.prepare(h)
    attn_out = self_attn(h_conv)          (q/k norm, rope, attn vs prefix+block)
    attn_out = attention_conv.finish(attn_out, coeff)
    residual = x; x = attn_out
    h2 = post_attention_layernorm(x [+ residual])
    h2c, mcoeff = mlp_conv.prepare(h2)
    mlp_out = mlp(h2c)
    mlp_out = mlp_conv.finish(mlp_out, mcoeff)
    x = mlp_out

Selector (dflash.py CandidateSelector + _score_edges, greedy):
    unary = topk(hidden @ lm_head[:VOCAB].T)
    scores[p, c] = unary[p, c] + <A[pred_p] * P(hidden_p), B[cand_p_c]>
    pred_0 = anchor codes; pred_p = cand_{p-1}
    greedy: argmax per slot (no branching).

Writes fixtures/sglang_golden.npz + shared fixture lm_head.npy.

Run: /mnt/ollama/models/glm-5.3-flash/sglang-venv/bin/python tests/golden/sglang_ref_dump.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

CHECKPOINT = "/mnt/ollama/models/glm-5.3-flash/dflash2/model.safetensors"
CONFIG = "/mnt/ollama/models/glm-5.3-flash/dflash2/config.json"
HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
TARGET_LAYERS_0IDX = [5, 14, 24, 33, 42]

BLOCK_SIZE = 8   # dflash_config.block_size
VOCAB = 154880   # glm5next.vocab_size
GROUP = 16       # conv_group_size


def rms_norm(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def grouped_conv(x, base, delta, n_groups, group_size, taps):
    """dflash.py _grouped_conv for one block: x [B,S,H], delta [B,S,taps,groups],
    base [taps, H] (viewed to [taps, groups, group_size]). Block-relative positions."""
    blocks = x.float().unflatten(-1, (n_groups, group_size))          # [S,g,gs]
    coefficients = base.view(1, taps, n_groups, group_size) + delta.unsqueeze(-1)
    out = coefficients[:, 0] * blocks
    position = torch.arange(x.shape[0])  # S — x is [S, H], NOT [B,S,H]
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))  # slice seq axis (dim 0)
        out = out + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return out.flatten(-2)


def rope_cos_sin(pos, dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    freqs = pos.float()[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(q, cos, sin):
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    return torch.cat([q1 * cos[..., :half] - q2 * sin[..., :half],
                      q1 * sin[..., :half] + q2 * cos[..., :half]], -1)


def main():
    torch.manual_seed(42)
    from safetensors.torch import load_file
    ckpt = load_file(CHECKPOINT)
    with open(CONFIG) as f:
        cfg = json.load(f)

    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    eps = cfg["rms_norm_eps"]
    kvh = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    nh = cfg["num_attention_heads"]
    rep = nh // kvh
    theta = cfg["rope_parameters"]["rope_theta"]
    win = cfg["sliding_window"] - 1  # HF inclusive → window_left
    top_k = cfg["dflash_config"]["selector_top_k"]
    mask_id = cfg["dflash_config"]["mask_token_id"]
    n_groups = H // GROUP
    kv_size = kvh * hd

    fz = np.load(os.path.join(FIXTURES, "hiddens.npz"))
    tokens_np = fz["tokens"]
    layers = [torch.from_numpy(fz[f"layer_{il + 1}"]).float() for il in TARGET_LAYERS_0IDX]
    N = tokens_np.shape[0]
    print(f"ckpt L={L} H={H} | fixture N={N} | rope_theta={theta} win={win}")

    # ---- 1. project_target_hidden ----------------------------------------
    ctx_hidden = F.linear(torch.cat(layers, -1), ckpt["fc.weight"].float())
    ctx_hidden = rms_norm(ctx_hidden, ckpt["hidden_norm.weight"], eps)
    print("ctx_hidden:", tuple(ctx_hidden.shape))

    # ---- 2. prefix KV (kv_proj_only + k_norm + k_rope per layer) ---------
    positions_ctx = torch.arange(N)
    kv_k, kv_v = [], []
    for il in range(L):
        p = f"layers.{il}.self_attn."
        kv = F.linear(ctx_hidden, torch.cat(
            [ckpt[p + "k_proj.weight"], ckpt[p + "v_proj.weight"]]).float())
        k, v = kv.split([kvh * hd, kvh * hd], -1)
        k = k.reshape(N, kvh, hd)
        k = rms_norm(k, ckpt[p + "k_norm.weight"], eps)
        cos, sin = rope_cos_sin(positions_ctx, hd, theta)
        k = apply_rope(k, cos[:, None, :], sin[:, None, :])
        kv_k.append(k)
        kv_v.append(v.reshape(N, kvh, hd))
    print("prefix KV done")

    # ---- 3. shared embed/lm_head fixture (both arms load this) -----------
    head_path = os.path.join(FIXTURES, "lm_head.npy")
    if os.path.exists(head_path):
        emb_w = torch.from_numpy(np.load(head_path)).float()
    else:
        g = torch.Generator().manual_seed(123)
        emb_w = torch.randn(VOCAB, H, generator=g) / np.sqrt(H)
        np.save(head_path, emb_w.numpy())
        print("created shared embed fixture")

    # ---- 4. noise block + draft forward ----------------------------------
    anchor_id = int(tokens_np[-1])
    block_ids = torch.full((1, BLOCK_SIZE), mask_id, dtype=torch.long)
    block_ids[0, 0] = anchor_id
    S = BLOCK_SIZE
    x = emb_w[block_ids.flatten()].view(S, H).clone()
    residual = None
    pos_blk = torch.arange(N, N + S)
    # sliding window over prefix keys: window [pos - win, pos]
    pos_arr = pos_blk.float()

    hiddens_per_layer = []
    for il in range(L):
        p = f"layers.{il}."
        if residual is None:
            residual = x
            h = rms_norm(x, ckpt[p + "input_layernorm.weight"], eps)
        else:
            h = rms_norm(x + residual, ckpt[p + "input_layernorm.weight"], eps)
            residual = x + residual
        hiddens_per_layer.append(h.clone())

        # attention sublayer, conv-prepared
        aconv = ckpt[p + "attention_conv.base_kernel"].float()
        aproj = ckpt[p + "attention_conv.kernel_projection.weight"].float()
        coeff = F.linear(h, aproj).reshape(S, 2, 2, n_groups)
        h_conv = grouped_conv(h, aconv[0], coeff[..., 0, :, :], n_groups, GROUP, 2)

        # attention
        q = F.linear(h_conv, ckpt[p + "self_attn.q_proj.weight"].float()).reshape(S, nh, hd)
        kk = F.linear(h_conv, ckpt[p + "self_attn.k_proj.weight"].float()).reshape(S, kvh, hd)
        vv = F.linear(h_conv, ckpt[p + "self_attn.v_proj.weight"].float()).reshape(S, kvh, hd)
        q = rms_norm(q, ckpt[p + "self_attn.q_norm.weight"], eps)
        kk = rms_norm(kk, ckpt[p + "self_attn.k_norm.weight"], eps)
        cos, sin = rope_cos_sin(pos_blk, hd, theta)
        cos = cos[:, None, :]; sin = sin[:, None, :]
        q = apply_rope(q, cos, sin)
        kk = apply_rope(kk, cos, sin)

        qf = q.permute(0, 1, 2).transpose(0, 1)                       # [nh, S, hd]
        kf_blk = kk.permute(1, 0, 2)                                  # [kvh, S, hd]
        vf_blk = vv.permute(1, 0, 2)
        # prefix keys with sliding-window mask per query position
        kpre = kv_k[il].permute(1, 0, 2)                              # [kvh, N, hd]
        vpre = kv_v[il].permute(1, 0, 2)
        kpre = kpre.repeat_interleave(rep, 0)
        vpre = vpre.repeat_interleave(rep, 0)
        kf_blk = kf_blk.repeat_interleave(rep, 0)
        vf_blk = vf_blk.repeat_interleave(rep, 0)

        att = torch.empty(nh, S, hd)
        for si in range(S):
            qpos = int(pos_blk[si])
            lo = max(0, qpos - win)
            k_full = torch.cat([kpre[:, lo:qpos + 1], kf_blk], 1)     # windowed prefix + block
            v_full = torch.cat([vpre[:, lo:qpos + 1], vf_blk], 1)
            scores = torch.einsum('hd,hld->hl', qf[:, si], k_full) / (hd ** 0.5)
            att[:, si] = torch.einsum('hl,hld->hd', scores.softmax(-1), v_full)
        attn_out = att.transpose(0, 1).reshape(S, nh * hd)
        attn_out = F.linear(attn_out, ckpt[p + "self_attn.o_proj.weight"].float())
        attn_out = grouped_conv(attn_out, aconv[1], coeff[..., 1, :, :], n_groups, GROUP, 2)

        residual = residual + 0  # (unchanged)
        x = attn_out

        h2 = rms_norm(x + residual, ckpt[p + "post_attention_layernorm.weight"], eps)
        mconv = ckpt[p + "mlp_conv.base_kernel"].float()
        mproj = ckpt[p + "mlp_conv.kernel_projection.weight"].float()
        mcoeff = F.linear(h2, mproj).reshape(S, 2, 2, n_groups)
        h2c = grouped_conv(h2, mconv[0], mcoeff[..., 0, :, :], n_groups, GROUP, 2)
        gate = F.linear(h2c, ckpt[p + "mlp.gate_proj.weight"].float())
        up = F.linear(h2c, ckpt[p + "mlp.up_proj.weight"].float())
        act = F.silu(gate) * up
        mlp_out = F.linear(act, ckpt[p + "mlp.down_proj.weight"].float())
        mlp_out = grouped_conv(mlp_out, mconv[1], mcoeff[..., 1, :, :], n_groups, GROUP, 2)
        x = mlp_out

    h_final = rms_norm(x + residual, ckpt["norm.weight"], eps)       # [8, H]
    h_final = h_final
    print("h_final:", tuple(h_final.shape))

    # ---- 5. selector lattice + greedy walk -------------------------------
    pred_hidden = h_final[1:]                                        # [7, H]
    logits = pred_hidden @ emb_w[:VOCAB].T
    topv, cand = logits.topk(top_k, dim=-1)                          # [7, K]
    hid_proj = F.linear(pred_hidden, ckpt["candidate_selector.hidden_projection.weight"].float())
    A = ckpt["candidate_selector.predecessor_codebook"].float()
    B = ckpt["candidate_selector.successor_codebook"].float()
    keys = B[cand]                                                   # [7, K, rank]
    pred_ids = torch.cat([torch.full((1, top_k), anchor_id, dtype=torch.long),
                          cand[:-1]], dim=0)                        # [7, K]
    preds = A[pred_ids]                                              # [7, K, rank]
    scores = topv + torch.einsum(
        "bkr,bkr->bk",
        preds * hid_proj[:, None, :],
        keys,
    )
    # greedy walk (sample_path with greedy_mask: argmax per slot row)
    path = [scores[0].argmax()]
    for e in range(1, S - 1):
        path.append(scores[e].argmax())
    proposed = cand.gather(-1, torch.stack(path, -1)[:, None])[:, 0]

    np.savez(os.path.join(FIXTURES, "sglang_golden.npz"),
             tokens=tokens_np, ctx_hidden=ctx_hidden.numpy(),
             h_final=h_final.numpy(), candidate_ids=cand.numpy(),
             unary_logits=topv.numpy(), lattice_scores=scores.numpy(),
             proposed_ids=proposed.numpy(), anchor_id=np.int64(anchor_id))
    print("wrote sglang_golden.npz")
    print("proposed:", proposed.numpy().tolist())


if __name__ == "__main__":
    main()
