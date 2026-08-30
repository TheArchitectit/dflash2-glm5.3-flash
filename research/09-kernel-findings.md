# Kernel findings — verify cost curve & MoE routing measurement (2026-08-30)

Program: `openspec/changes/add-moe-kernel-tuning/` (REQ-K1..K4). Box ucs03
(40-core dual-Xeon, ~120 GB/s, no GPU), GLM-5.3-Flash IQ4_XS + DFlash2 f16
draft, probe build = fork copy `llama-cpp-kernel` @ `kernel/moe-probe`.

## W1 — verify cost curve (synth full-acceptance)

ctx 16384, one fixed prompt, 3 reps/point, warm page cache; cycle_s =
(n_max+1)/predicted_per_second from server timings.

| verify tokens | cost (s) | reps |
|---|---|---|
| 1 (spec-off anchor, Sprint 5.1) | 0.758 | — |
| 5 (n_max=4, synth 1,1,1,1) | **1.391** | 1.361/1.432/1.391 |
| 8 (n_max=7, synth 1..1 ×7) | **1.866** | 1.898/1.849/1.866 |

Two-point fit `cost(n) = a + b·(n−1)`:

- **b = 0.158 s per extra verify token = 0.21× single-token** (gate "flat": <0.3×)
- **a = 0.759 s — free fit lands on the measured single-token cost 0.758**
  (0.1% agreement): weight traffic is effectively amortized across the
  batch; the batch does NOT re-read weights per token.
- kill-check vs Sprint 5.1's 2.09 s at n=8: we measured 1.87 (−11%).
  Difference is explained by ctx (16k here vs 131k then) — the residual b
  term is partly ctx-dependent (attention/DSA), so 2.75× was the
  production-ctx number; 2.46× at 16k. Curve is linear in both regimes.

### Consequence for F3 (expert-reuse-aware verify)

REQ-K1 branch table: **flat curve ⇒ the MoE-dedup lever is at the gate
threshold**, pending W3's MoE perf-share. The cost decomposition already
says duplicate `vec_dot`s can't hold the b-term: a(=single-token weight
cost) + b·n(=per-token marginal compute) fit the data with zero
weight-scaling term.

## W2 — routing probe (LLAMA_MOE_PROBE=1, thread-0 counters, GLM target)

From the atexit dumps (`benchmarks/raw/k1_win1_n7.log`, `k1_win1_n4.log`,
extract `moe_probe_win1.txt`):

| n_tok/call | calls | pairs/call | touched/call | col_rate | cne1: 1 / 2 / 3 / 4-8 |
|---|---|---|---|---|---|
| 2 | 252 | 16 | 15.1 | 0.058 | 3564 / 234 / — / — |
| 5 (n4-verify) | 4536 | 40 | **27.6** | 0.309 | 90354 / 21780 / 7659 / 5589 |
| 8 (n7-verify) | 2646 | 64 | **38.3** | 0.402 | 65796 / 19608 / 7800 / 8022 |

- **Adjacent-draft routing correlation is strong**: 8-token verify touches
  38.3 distinct experts vs RC-1's uncorrelated floor of 57.9 (−34%). RC-1's
  "~6.5× the FFN weight read" was an overestimate; measured ~4.8× vs a
  single token's 8 — and per W1 that cost is paid once (amortized into a),
  not per token.
- Intra-token collision baseline: the n2 row (col_rate 0.058) is the
  within-one-token floor — grouped top-k can repeat an expert among a
  token's 8 choices. The verify-relevant signal is the EXCESS above this
  baseline: +0.34 at n8 (same convention as `moe_probe_parse.py`'s
  `col_rate_excess_vs_n1`, calibrated on Qwen3-Coder-Next where the floor
  is 0.43 with 16-of-512 grouped routing).
- cne1>8 buckets are empty: no expert ever receives >8 of the 64 slots.

### Reading

The remaining b·n = 0.158 s/token is per-token marginal cost, not expert
weight traffic (which W1 shows amortized). Candidates: KDA recurrent-state
update (34 layers — ctx-dependent component matches the Sprint-5.1 Δ),
attention, activations, dequant ALU. **W3 (perf attribution) decides which.**
If MoE (mul_mat_id/vec_dot) share ≥ 35% despite the flat curve, the lever
that survives is IQ4_XS **repack coverage** (branch b — speed up dequant,
not dedup), not expert-reuse. If KDA/ssm dominates → branch c.

## Gate status

REQ-K1 decision table: flat curve (b=0.21× < 0.3×) → branch (a) expert-reuse
**retired pending W3 confirmation**; (b) IQ4_XS repack and (c) KDA batching
live, decided by the perf shares in W3.

## W3 — op-class attribution (perf record LBR, synth n_max=7, decode-only)

Captured after the 147 GB load so samples exclude startup; 2.84 M samples,
`ggml_graph_compute_thread` = 99.56% (thread pool is where all CPU work is).
Self% of the hot leaves (children/inclusive overlap — self is the honest
metric):

| symbol | self% | what |
|---|---|---|
| `ggml_vec_dot_iq3_s_q8_K` | **32.89%** | IQ3_S dot — MoE routed experts (82/129 tensors) |
| `ggml_compute_forward_mul_mat` | 17.67% (incl) | dense/attn matmul wrapper (→ sgemm, Q8_0 shared experts) |
| `llamafile_sgemm` / tinyBLAS_Q0_AVX | 13.21% / ~10% (incl) | Q8_0/F16 GEMM (shared experts, dense) |
| `ggml_vec_dot_iq4_xs_q8_K` | **5.88%** | IQ4_XS dot — 41/129 routed tensors |
| `ggml_vec_dot_q6_K_q8_K` | 1.91% | Q6_K dot |
| `ggml_compute_forward_gated_delta_net` (KDA) | **1.04%** | the 34 linear-attn layers |
| `flash_attn_ext` | 0.74% | attention |

### Branch decision (W3 overrides the W1 gate's tentativeness)

- **Branch (c) KDA batching — DEAD.** gated_delta_net is 1.04%; even making
  it free is ~1%. The 0.158 s/token b-term is not KDA.
- **Branch (a) expert-reuse batching — bounded out.** Amdahl on the flat
  curve: making the 61% of pairs that sit in multi-token expert buckets
  *free* saves ≤ 0.61 × (0.3289+0.0588) = 23.7% raw, but the W1 fit says
  that traffic is already L1-amortized (a = single-token cost). Realistic
  fused-decode speedup (load IQ3_S block once for k tokens, ~1.5–2× on the
  decode overhead of the multi-token 35% of buckets) lands **8–12%**, under
  the ≥15% gate.
- **Branch (b′) mixed-IQ microkernel — the surviving lever, and bigger
  than planned.** The single hottest function in the entire server is
  `ggml_vec_dot_iq3_s_q8_K` (32.9% self), and it is **workload-agnostic**:
  it dominates single-token decode (the 1.32 t/s baseline), prefill, AND
  verify. A faster IQ3_S dot (the x86 tree has one AVX2 path at
  `arch/x86/quants.c:3384`; RISC-V already carries vl128/256/512
  specializations the x86 build does not) lifts every phase at once —
  which is what a ≥15% *effective* win on a bandwidth-bound box actually
  requires. The planned "IQ4_XS repack" was aimed at the wrong quant:
  IQ4_XS is only 5.9% here.

### Next: K-4 branch (b′) — design before code

A read-only design pass (cpp-pro specialist) is checking whether IQ3_S /
IQ4_XS drop into the `repack.cpp` `tensor_traits<BLOC,INTER,NB_COLS>`
template or need a bespoke AVX-512/VNNI microkernel, and whether
`ggml_vec_dot(..., nrc)` multi-column paths exist (every quants.c
implementation asserts `nrc == 1`, so multi-token fusion must be a new
kernel, not a flag). That determines "add traits" vs "write kernel".

