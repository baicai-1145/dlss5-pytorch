# Oracle Probe Findings — Round 1 (flat)

Date: 2026-09-04 night · Source: `dlssnr-probe-flat` (SyntheticFeed, 5 flat frames cycled, probe=1 verified)

## Setup

SyntheticFeed replaced the game frame with flat gray levels (luma 0/0.25/0.5/0.75/1.0
linear HDR). Capture recorded the full chain: encode (before) → DLL evaluate (model_raw).

## Official model response to flat inputs (the U-curve, measured)

| linear luma | before (sRGB) | delta R/G/B (model_raw − before) | spatial std |
|---|---|---|---|
| 0.00 | 0.000 | +0.0000/+0.0007/+0.0000 (≈zero) | 0.0003 |
| 0.25 | 0.537 | +0.0043/+0.0025/+0.0011 (lift) | 0.0010 |
| 0.50 | 0.735 | −0.0031/−0.0013/−0.0008 (press) | 0.0028 |
| 0.75 | 0.880 | −0.0026/−0.0023/−0.0027 (press) | 0.0014 |
| 1.00 | 0.958 | −0.0084/−0.0070/−0.0077 (press hard) | 0.0020 |

## Conclusions

1. **U-shaped brightness adaptation confirmed with exact numbers** — shadows lifted,
   highlights pressed. Matches the luma-binned DC analysis from gameplay captures.
2. **Edits on noise-free flat frames are tiny (≤0.008) and spatially uniform** —
   the model is near-identity without noise. The ~28 dB-scale deltas seen in gameplay
   are the *noise response*, not unconditional regeneration.
3. **Deterministic oracle** — same luma → same delta across cycles (±0.0005).
   These five rows are now per-point acceptance targets for the replica.

## Next

- impulse/comb/grayramp/reset probes → receptive field, frequency response,
  full transfer curve, temporal convergence trace.
- Replica runs the same probes; deviations localize which module is mis-loaded
  (U-curve = global path, impulse spread = receptive field, reset trace = recurrence).
