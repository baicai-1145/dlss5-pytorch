# R30 Track B — frida hook specification (only if Track A probes are inconclusive)

Goal: capture, at runtime, which texture descriptor id 0x6e is bound to
inside the `cc_tinlayout_fused_post_block_swin_1h_32_simple_blend*`
kernels — i.e. what resource the MV-bicubic residual actually samples.

## Hook points

1. **`cuLaunchKernel` / `cuLaunchKernelEx`** (driver API, nvcuda.dll)
   - Filter: kernel function name in
     `{cc_tinlayout_fused_post_block_swin_1h_32_simple_blend,
       ..._simple_blend_full_rect, ..._simple_blend_fp8,
       ..._simple_blend_fp8_full_rect}`
   - Grab `void** kernelParams` (or the extra blob for Ex) at launch.

2. **Param extraction** — from R21 the CONSTANT[0] window holds the
   launch-time params at c[0x60]/c[0x68]/c[0x90]/c[0x98]/c[0x168]/c[0x180]/c[0x184].
   At the launch call the param area is what lands in constant bank 0:
   - Dump the full param buffer (walk kernelParams until the known layout
     size; simple_blend layout ≈ 0x1E8 bytes, 7 descriptors of 8 bytes each
     + scalars).
   - The CUDA texture descriptors are `CUDA_TEXTURE_DESC`-backed handles;
     log each 8-byte slot: the odd slots (0x5a→0x66→0x6e→0x70→0x90) map to
     sampler/texture pairs — record `slot → CUtexObject` value.

3. **`cuTexObjectGetResourceDesc` / `cuTexObjectGetTextureDesc`** on each
   captured CUtexObject:
   - `resType == CU_RESOURCE_TYPE_PITCH2D/ARRAY`: log array dimensions +
     format. Cross-match with the resources we know: encode input
     (R10G10B10A2 1920×1050), model output planes (4 separate R10G10B10A2
     planes from outview — R27), motion (R16G16), depth (R32F).
   - **The 0x6e slot's resource dims/format identify it unambiguously**:
     - dims = full-res single plane, R10G10B10A2, matches `g_nr.colorCopy`
       → H_A (input color)
     - dims = full-res, but created AFTER the network kernel of the same
       frame (compare `CUresource` creation order or just the pointer
       against the output-plane pointer logged at outview launch)
       → H_B (prev network output)

4. **Cross-check anchor**: also capture the `outview` kernel launch and
   log its 4 output plane device pointers (params at c[0x168] base).
   If 0x6e's CUtexObject resolves to one of THESE pointers → H_B proven
   directly; if it resolves to the encode-input copy → H_A proven.

## Practical notes

- Driver API hooking: `frida-trace -i "cuLaunchKernel*"` works on the
  process; but we need param unpacking → use a JS script with
  `Interceptor.attach(Module.getExportByName('nvcuda.dll','cuLaunchKernel'))`.
- The DLL may use the runtime API (`cudaLaunchKernel`) instead — hook both.
- 5090 is sm_120: the cubins we decoded are from the shipped blob; confirm
  the runtime picks the same kernel names (log every launch name for 1
  frame, then filter).
- Dump per launch: kernel name, grid dim, 0x1E8 param bytes (hex), and
  `CUtexObject` → resolved `{resType, dims, format}` for slots 0x66/0x6e/0x70.
- One evaluate frame is enough (16 launches); capture while the game is on
  the probe feed (Track A shots running) so the resource identities are
  trivially stable.

## Decision tree

- 0x6e → encode input copy: **H_A confirmed** → replica full-tail with
  H_A becomes the calibrated sharpness carrier; refit gate scale on
  gameplay, rerun scoreboard.
- 0x6e → outview output plane: **H_B confirmed** → implement H_B source
  in DLSS5_TAIL_MODE=full (prev-output buffer), rerun scoreboard.
- 0x6e → neither (a third buffer, e.g. DLSS-internal ping-pong): log dims
  + pointer provenance and bring the numbers back; that would open a new
  lane (internal accumulation buffer) — new round needed.

## UPDATE (Windows recon): probe_queue path + bin format

- Queue dir: `C:/Control.Ultimate.Edition.v517.915/probe_queue/`; OptiScaler
  source (ff20fe47) at the game dir `.tmp/OptiScaler_DLSSNR/`.
- Existing bin sizes prove COLOR-ONLY packing: comb 32MB = 2 frames,
  grayramp 16MB = 1, impulse 258MB = 16, reset 129MB = 8 — exactly
  `16 bytes header + frames × 1920×1050×8 B` (RGBA16F). Header:
  `<u32 magic 0x31425250 'PRB1'><u32 W><u32 H><u32 frameCount>` then
  raw half-float frames. Depth/motion are NOT in the bin (game-side).
- **Probes therefore use TEMPORAL DISPLACEMENT** (pattern jumps between
  frames, game motion ≈ 0 on a frozen scene) instead of MV injection:
  - `capP3_impulse.bin` (4f): 8×8 dot (cx,cy) on f0 → (cx+64,cy) on f1-3.
    Ghost dot at the OLD position in the delta = H_B; new-position-only
    = H_A. Reset frame (f0) clean for both.
  - `capP2_edge.bin` (4f): vertical edge x=960 → x=1024. Ghost edge at
    the old position = H_B; band-width measurement for free.
  - `capP1_dc.bin` (2f): gray 0.25 → 0.75 DC step; gate/baseline calib.
- Generator (Mac/3090 side, produces the .bin files):
  `.tmp/gen_probe_mvmotion.py` — run, then drop the three bins into
  `probe_queue/` (one per capture session, per the queue's one-item-per-
  run semantics) and rename each dump dir `capP3_impulse` / `capP2_edge`
  / `capP1_dc`.
- Pending: Windows pi is confirming the exact frame-packing layout from
  the ff20fe47 source (color-only vs interleaved planes); if it differs
  from the PRB1 header+RGBA16F plan above, the generator gets a small
  packer patch — the PATTERNS themselves are layout-independent.
