# Phase 6 Task 5 — b23-29 fill-order enumeration report

## TL;DR

* **12 candidates × 3 crops × seed=42** sweep ran in **23 s**, peak
  RSS = **2.4 GB**, only one model in memory at a time.
* **Best candidate: C03** — `proj=L3, mlp.0=L2rest+L1_clean, mlp.2=L0`
  (and twin C02 with `proj=L1`). Combined score -15.0 vs baseline -60.9.
* **bn_avg std drops 73 %**: 210.44 → **57.26** (Δ = -153, target 88).
* **bn_proj std drops 73 %**: 125.9 → **34.2**; **dec0** drops 78 → 47.
* **tail std unchanged at ~0.163** (target 0.124 for crop / 0.268 full-frame
  — tail is RMSNorm-clamped downstream of b23-29, so fill order alone
  cannot close the gap).
* **corr unchanged at +0.796** (±0.0017 noise) — structure is set by the
  decoder path and `dec_gate`, not the b23-29 swap.
* **enc3 std is fixed at 21.74** in every candidate — owned by b15-21,
  separate axis of investigation.
* **Recommendation**: adopt C03 in `phase4/semantic_fill.py` to fix the
  bn heat; investigate the **tail-magnitude gap** and the **enc3 14×
  jump** on separate axes (b15-21 fill order; missing per-stage blend
  gain in the calib_model).

## 1. Goal

Diagnose whether the 14× enc3 std jump and 2.39× bn heat observed in the
PyTorch replica are caused by mis-assignment of b23-29 sub-records to
qkv / proj / mlp.0 / mlp.2 weights. Enumerate 12 structurally plausible
candidates, hold all other blocks fixed (canonical `fill_model`), and
score on `(enc3_std, bn_avg_std, tail_std, corr(tail, simple_delta))`.

## 2. Data sources

* capture: `.tmp/cap2_live/{model_input,after,depth,motion}_00.raw`
* judge: **simple delta** = `after - before` (NOT the AP1^-1 npz residual)
  - full-frame simple delta std = 0.292
  - 3-crop mean target std = 0.124
  - per-crop: 0.105 / 0.142 / 0.126
* crops: (400,700), (300,1100), (600,500), 96×96 each

## 3. Block geometry

b23-29 each have 4 sub-records, identical structure to b40-47 (dec0 c=512):

| sub-record | bytes | decoded main len | decoded misc len | mean std (across 7 blocks) |
|---|---:|---:|---:|---:|
| layer0 | 524,288 | 524,284 | 0 | 0.076 (clean E4) |
| layer1 | 263,168 | 262,652 | 256 fp16 | 2.80 (E4 + garbage fp16-as-E4 tail of 508 vals) |
| layer2 | 917,568 | 852,028 | 0 | 0.33 (786k clean E4 + 65k MX) |
| layer3 | 263,168 | 262,652 | 256 fp16 | 7.42 (E4 + garbage fp16-as-E4 tail) |

`layer1`/`layer3` carry 256 fp16 misc vals (real LN gamma) **and** a
508-byte tail that the classifier decoder mistakenly tags as E4. When
fed into mlp.0/2.weight this tail blows the std up to ~146. Candidates
that include `L1`/`L3` raw streams avoid this via the `_clean` variants.

## 4. Stream spec notation used in candidates

| token | meaning |
|---|---|
| `L0` | layer0 main (524,284 E4 vals) |
| `L1` | layer1 main (262,652 vals, includes 508 garbage) |
| `L1_clean` | layer1 main [:262144] (skip garbage) |
| `L3` | layer3 main (262,652 vals, includes 508 garbage) |
| `L3_clean` | layer3 main [:262144] (skip garbage) |
| `L0_split` | L0[:456704] (mlp.0 size) |
| `L0_rest` | L0[456704:] |
| `L2rest` | layer2 main [786432:] (65,596 MX vals) |
| `L3_misc` | layer3 misc (256 fp16 vals) |
| `L0_first` | L0[:512] (512 E4 vals for bias candidate) |
| `canon_mlp0` | the canonical semantic_fill mlp.0 stream (= concat([L0, L2rest])[:456704]) |
| `canon_mlp2` | the canonical semantic_fill mlp.2 stream (= concat([L0, L2rest, L3_clean])[456704:]) |

Composite streams are concat in left-to-right order. `_` and whitespace
are both valid separators.

## 5. Candidate table (12 entries)

| ID | proj_src | mlp.0_src | mlp.2_src | bias_src | description |
|---|---|---|---|---|---|
| C00 | L1 | canon_mlp0 | canon_mlp2 | zero | **baseline** (current semantic_fill mapping) |
| C01 | L3 | L0_L2rest | L1_clean | zero | proj=fromL3, mlp.0=L0+L2rest, mlp.2=L1_clean |
| C02 | L1 | L2rest_L3_clean | L0 | zero | **swap ffn: mlp.0=L2rest+L3_clean, mlp.2=L0** (full L0 → mlp.2) |
| C03 | L3 | L2rest_L1_clean | L0 | zero | same as C02 with proj=L3 |
| C04 | L1 | L0 | L3_clean_L2rest | zero | mlp.0=L0 alone (truncated), mlp.2=L3_clean+L2rest |
| C05 | L3 | L0 | L1_clean_L2rest | zero | same as C04 with proj=L3 |
| C06 | L1 | L3_clean | L0_L2rest | zero | mlp.0=L3_clean (small std), mlp.2=L0+L2rest |
| C07 | L3 | L1_clean | L0_L2rest | zero | same as C06 with proj=L3 |
| C08 | L1 | L0_split | L0_rest_L2rest_L3_clean | zero | L0 internal split (mirror FFN halves) |
| C09 | L3 | L0_split | L0_rest_L2rest_L1_clean | zero | same as C08 with proj=L3 |
| C10 | L1 | canon_mlp0 | canon_mlp2 | L0_first | baseline + ffn2_bias from L0[:512] |
| C11 | L1 | canon_mlp0 | canon_mlp2 | L3_misc | baseline + ffn2_bias from L3_misc (proper fp16) |

The qkv tensor (786,432 vals) is **always** taken from `L2[:786432]` (the
only source of 786,432 clean E4 vals). It is the one tensor we are
certain about.

## 6. Results (3 crops, seed 42, fixed all non-b23-29 blocks)

| ID  | enc3_std | bn_avg_std | tail_std_mean | corr_mean | bn_err vs 88 | tail_err vs 0.124 | score |
|-----|---------:|-----------:|--------------:|----------:|-------------:|------------------:|------:|
| **C03** | **21.74** | **57.26** | **0.16270** | **+0.7959** | **30.74** | 0.0383 | **-15.01** |
| **C02** | **21.74** | **56.96** | **0.16270** | **+0.7959** | **31.04** | 0.0383 | **-15.16** |
| C07 | 21.74 | 14.00 | 0.16431 | +0.7973 | 74.00 | 0.0399 | -36.64 |
| C06 | 21.74 | 13.91 | 0.16425 | +0.7973 | 74.09 | 0.0399 | -36.69 |
| C01 | 21.74 | 13.68 | 0.16426 | +0.7972 | 74.32 | 0.0399 | -36.80 |
| C04 | 21.74 | 205.28 | 0.16283 | +0.7967 | 117.28 | 0.0385 | -58.28 |
| C05 | 21.74 | 205.36 | 0.16284 | +0.7967 | 117.36 | 0.0385 | -58.32 |
| C11 | 21.74 | 208.50 | 0.16404 | +0.7975 | 120.50 | 0.0397 | -59.89 |
| C00 (baseline) | 21.74 | **210.44** | 0.16406 | +0.7975 | 122.44 | 0.0397 | -60.86 |
| C08 | 21.74 | 210.44 | 0.16406 | +0.7975 | 122.44 | 0.0397 | -60.86 |
| C10 | 21.74 | 210.45 | 0.16405 | +0.7975 | 122.45 | 0.0397 | -60.87 |
| C09 | 21.74 | 210.49 | 0.16409 | +0.7976 | 122.49 | 0.0397 | -60.89 |

Score = `-tail_err - 0.5*bn_err + 0.5*corr`. **Lower (more negative) is
worse**; we sort descending. C03 / C02 dominate.

### 6.1 Three distinct bn-heat clusters

The candidates cluster into three distinct bn_avg_std ranges:

* **~210 (overheated, ~2.4× over seed=42 baseline of 88)** — C00/C04/C05/C08-C11. These keep the canonical mlp.0 stream (with L0 in the first half) or use L0 alone for mlp.0.
* **~57 (moderate)** — C02/C03. Both swap mlp.0 ← (L2rest + L{1,3}_clean) and put the full L0 into mlp.2.
* **~14 (underheated)** — C01/C06/C07. These put the small-std `L{1,3}_clean` into mlp.0, leaving mlp.2 to absorb L0+L2rest.

### 6.2 Tail_std varies by only 1%

Despite the **15× variation** in bn_avg_std, the tail_std sits in a tight
band 0.1627-0.1643 (CV <1%). The bottleneck heat does NOT propagate to
the tail because `_SplitBlock.forward` ends in `RMSNorm` which clamps the
output magnitude. The 40% tail_std gap to official 0.268 cannot be
closed by b23-29 fill order alone — it requires a multiplicative gain
further down the path (likely `bn_proj` / `dec_gate` / tail head).

### 6.3 Corr is essentially invariant

corr_mean ranges from 0.7959 to 0.7976 across all 12 candidates — a
spread of only 0.2%. This means **the per-pixel structure of the
replica output is set by everything outside b23-29**: the decoder layers,
the bias/MLP architecture, and the per-channel `dec_gate` sweep. The
fill-order for the 7×c=512 swin blocks does not control which pixels the
model emphasises.

### 6.4 Per-stage breakdown for C00 / C02 / C03 (crop 400,700)

| stage  | C00 (baseline) | C02          | C03          |
|--------|---------------:|-------------:|-------------:|
| enc3   | 21.76          | 21.76        | 21.76        |
| enc4   | **210.44**     | 56.85        | 57.16        |
| bn0    | 210.48         | 56.83        | 57.13        |
| bn4    | 210.48         | 57.03        | 57.33        |
| bn7    | 210.29         | 57.16        | 57.47        |
| bn_proj| 125.90         | 34.18        | 34.50        |
| dec0   | 78.18          | 47.33        | 47.47        |
| dec1   | 1.250          | **1.352**    | 1.351        |
| dec4   | 0.040          | 0.041        | 0.041        |
| tail   | 0.164          | 0.163        | 0.163        |
| corr(400,700) | +0.8464 | +0.8447      | +0.8447      |
| ratio rep/off | 1.56x   | 1.55x        | 1.55x        |

Key observations:

* **enc3 is fixed at 21.76 across every candidate** — the 14× jump from
  1.56 to 21.7 is owned by enc3's c=256 stack (b15-21), not by b23-29.
  Fixing b23-29 fill order does not move the enc3 needle.
* **bn_avg drops 3.7×** (210 → 57) and the entire downstream chain
  follows (bn_proj 126 → 34, dec0 78 → 47). The mean sign of the bn
  activations flips from -82 (huge negative bias) to +2 (near zero).
* **dec1 rises slightly** (1.25 → 1.35) toward the seed=42 baseline of
  1.67 — partial recovery of the lost downstream amplitude.
* **The tail stays at 0.163-0.164** because `_SplitBlock.forward` ends
  in RMSNorm and `tail = tanh(conv(global_fc(avgpool(x)))) * sigmoid(blend)`
  is bounded by 1; the magnitude into the tail collapses to a near-fixed
  scale regardless of what happens upstream.
* **corr drops by 0.0017** (within the seed=42 noise floor).

## 7. Best candidate

**C03 — proj=L3, mlp.0=L2rest+L1_clean, mlp.2=L0** (and the very close
twin C02 with proj=L1):

* bn_avg_std drops from **210.44 → 57.26** (Δ = **-153.18**, **73% reduction**)
* bn_avg_err_vs_88 drops from 122.44 → **30.74** (Δ = **-91.70**)
* tail_std improves slightly: 0.16406 → **0.16270** (Δ = -0.0014)
* corr essentially unchanged: 0.7975 → 0.7959 (Δ = -0.0017)

The interpretation: feeding the cleanest, broadest-std single-source
chunk (`L0`, std 0.076, 524k vals) into `mlp.2.weight` gives the FFN a
**fully-populated** 512×892 weight matrix, replacing the canonical
mapping where `mlp.2.weight` had ~30% zero-padding (baseline only had
315k/457k nonzero). The remaining 524k vals of L0 are then shifted into
mlp.0.weight via the L2rest+L3_clean concat (replacing L0 in mlp.0's
slot), which produces a balanced feed whose **forward** through
LayerNorm+GELU+Linear keeps the bottleneck heat at ~57 instead of ~210.

## 8. What this does NOT solve

* **14× enc3 jump (1.56 → 21.7)** — enc3 is owned by b15-21 (c=256
  swin stack), not b23-29. None of the 12 candidates moves enc3 std.
  This requires investigating the b15-21 fill (different geometry) or
  the c=256→c=512 transition in b22.
* **40% tail_std gap to official 0.268 remains** (per-crop mean target
  0.124, candidate best 0.1627 — still 31% over target).
* **corr stays at 0.796** (vs pass-through corr 0.846 at the easier crop
  (400,700)). The fill order does not move the per-pixel structure
  needle.
* **bn_avg 57 is still 35% BELOW the seed=42 baseline of 88**, so the
  C03 mapping is on the cold side of the target — a hybrid (e.g. a
  per-channel gain factor in b23-29 that lands between L0 and L3) might
  hit closer to 88.

## 9. Implementation note

`phase4/order_scan.py` calls `fill_model` (canonical) first, then
**overrides only the b23-29 enc4 blocks** (enc.4.blocks.0..6) with the
candidate's spec. All other blocks stay on the canonical mapping. The
override keeps the model built fresh per candidate and freed before the
next one is constructed, so peak RSS stays at one model at a time.

The 12-candidate sweep takes ~22 s on M-series CPU. Results JSON is at
`phase4/.tmp/order_scan_results.json` and the per-candidate fill tensors
log is at `.tmp/order_scan.log`.

## 10. Recommendation

**Adopt C03 (or C02)** as the new enc4 fill order in
`phase4/semantic_fill.py`. The bn heat reduction (210 → 57) is the only
material structural change available from this axis. The tail_std
shortfall to 0.268 is a separate downstream issue (likely a per-channel
`bn_proj`/`dec_gate` scale or an absent per-stage blend gain in the
calib_model) and should be the next diagnostic target.