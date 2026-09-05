// ============================================================================
// SyntheticFeed.h — R30 probe feed: MV-pattern generators for the 0x6e
// binding discrimination (H_A: bicubic source = current input color vs
// H_B: bicubic source = previous network output).
//
// Drop-in for the dlssnr-probe-flat branch (ff20fe47): replaces the game
// frame + motion vectors with deterministic synthetic patterns before the
// DLL evaluate. Same hook point as the flat probes (docs/PROBE_FINDINGS.md).
//
// Three probe families (run each as its own 4-frame shot):
//
//   P1 "flatmv"   — flat field luma 0.5 + CONSTANT full-screen MV (dx,dy)
//                   in {-0.25, -0.5, -1.0} px. H_A vs H_B differ wherever
//                   the bicubic warp of the source lands off-grid: for a
//                   flat field both sources sample identical values, so
//                   P1 alone only calibrates the warp magnitude. Its real
//                   job: verify the MV plumbing (delta must vanish if the
//                   kernel uses MV-warp at all, else MV is ignored).
//
//   P2 "edgeMV"   — vertical hard edge (left 0.3 / right 0.7 luma) +
//                   constant horizontal MV as above. THE discriminator:
//                   H_A samples the SHARP input edge at x+dx (sharp edge
//                   shifted by exactly dx), H_B samples the NETWORK OUTPUT
//                   (edge already smoothed by the swin chain + previous
//                   blend), so the sampled profile differs in the
//                   2-3 px transition band. Measure delta transition-band
//                   width + edge position shift:
//                     width(H_A) ~= input width (2 px hard edge stays ~2)
//                     width(H_B) ~= smoothed (4-6 px,网络已把边抹宽)
//                   Also: edge PEAK SHIFT = dx exactly for H_A (sampling
//                   input at +dx), while H_B shift tracks dx only on
//                   steady-state frames (prev output already moved).
//
//   P3 "impulseMV"— single bright dot (luma 1.0, 2x2 px) on black +
//                   constant MV, magnitudes {-0.5, -1.0} px. Sharpest
//                   possible kernel: H_A shows the dot moved by exactly
//                   (dx,dy) with bicubic ringing pattern of the INPUT
//                   dot (2 px); H_B shows the dot at prev-output position
//                   with the network's OWN smoothed footprint (r=8 spread,
//                   R28 radial fingerprint). Distance between delta peak
//                   and dot = dx for H_A on frame 0 (prev = 0 → bicubic
//                   reads zeros → NO moved dot visible in H_B on frame 0!).
//                   THE cleanest single-shot discriminator: on frame 0
//                   (reset=1), H_B's bicubic source is the prev output =
//                   zeros → delta shows NO dot displacement; H_A shows the
//                   dot displaced by exactly (dx,dy). One frame decides.
//
// Frames per shot: 4 (frame 0 = reset 1, frames 1-3 reset 0). The
// frame-0 discriminators are the cleanest (no history ambiguity);
// frames 1-3 measure steady-state behavior of both hypotheses.
//
// Output goes through the standard capture v3 dump (before/model_raw/
// hdr_copy/after/depth/motion + manifest). Naming:
//   capP1_mvp025, capP1_mvp050, capP1_mvp100,
//   capP2_mvp025, capP2_mvp050, capP2_mvp100,
//   capP3_mvp050, capP3_mvp100
// ============================================================================
#pragma once
#include <cmath>
#include <cstdint>

namespace probe
{
struct MVProbe
{
    float dx;   // motion vector x, PIXELS (engine convention: positive = content moved right)
    float dy;   // motion vector y, PIXELS
    int   pattern;  // 0=flatmv 1=edgeMV 2=impulseMV
};

// Call once per frame BEFORE the DLL evaluate, instead of copying the game
// frame into the encode input. Fills the R10G10B10A2 encode input and the
// R16G16 motion resource with the synthetic pattern.
//
//   frameIdx   : 0..3 within the shot
//   width/height : encode input dimensions
//   pBits      : mapped pointer of the encode input (R10G10B10A2, UINT32 per px)
//   pMotion    : mapped pointer of the motion resource (R16G16 float16 pair)
//   motionRowPitch : motion row pitch in bytes
//
// R10G10B10A2 packing (matches the game's encode format, linear HDR scale):
//   R = low 10 bits, G = mid, B = high 10, A = top 2 (set 0x3 = 1.0).
//
// Motion resource: the DLL reads HALF floats (u,v). The engine writes
// UV in its own convention; our replica applies U=-0.14 / V=+1.12 scaling
// (AGENTS.md). To make the DLL see a (dx,dy)-pixel motion, write:
//   u_half = dx / (-0.14)   v_half = dy / (+1.12)
// (the same inverse of the replica-side scaling — keeps replica and DLL
// on identical pixel conventions).
inline void fillProbeFrame(unsigned int frameIdx, unsigned int width, unsigned int height,
                           std::uint32_t* pBits, std::uint16_t* pMotion, unsigned int motionRowPitch,
                           const MVProbe& pr)
{
    const float lumaL = 0.3f, lumaR = 0.7f;   // P2 edge levels
    const float dotLuma = 1.0f;               // P3 dot level
    const unsigned int cx = width / 2, cy = height / 2;

    for (unsigned int y = 0; y < height; ++y)
    {
        std::uint32_t* row = pBits + (std::size_t)y * width;
        for (unsigned int x = 0; x < width; ++x)
        {
            float v = 0.0f;
            switch (pr.pattern)
            {
                case 0: v = 0.5f; break;                                   // flatmv
                case 1: v = (x < width / 2) ? lumaL : lumaR; break;        // edgeMV
                case 2:                                                    // impulseMV
                    if (x >= cx && x < cx + 2 && y >= cy && y < cy + 2)
                        v = dotLuma;
                    break;
            }
            // linear HDR value -> 10-bit int (encode scale: 1.0 = 1020? use
            // the same scale the flat probes verified: value*1020)
            unsigned int iv = (unsigned int) std::min(1023.0f, v * 1020.0f + 0.5f);
            row[x] = iv | (iv << 10) | (iv << 20) | (0x3u << 30);
        }
    }

    // constant motion field
    const unsigned int mrow = motionRowPitch / sizeof(std::uint16_t);
    const auto f32tof16 = [](float f) -> std::uint16_t {
        // IEEE half encode (values here are small, no denormal/overflow care)
        std::uint32_t x; std::memcpy(&x, &f, 4);
        std::uint32_t sign = (x >> 16) & 0x8000u;
        std::int32_t  e    = (std::int32_t)((x >> 23) & 0xFF) - 127 + 15;
        std::uint32_t m    = x & 0x7FFFFFu;
        if (e <= 0)  return (std::uint16_t)sign;
        if (e >= 31) return (std::uint16_t)(sign | 0x7C00u);
        return (std::uint16_t)(sign | ((std::uint32_t)e << 10) | (m >> 13));
    };
    const float u = pr.dx / (-0.14f);
    const float v = pr.dy / ( 1.12f);
    for (unsigned int y = 0; y < height; ++y)
    {
        std::uint16_t* mrowp = pMotion + (std::size_t)y * mrow;
        for (unsigned int x = 0; x < width; ++x)
        {
            mrowp[2 * x + 0] = f32tof16(u);
            mrowp[2 * x + 1] = f32tof16(v);
        }
    }
}

// Shot table — iterate these in the trigger handler (one shot = 4 frames,
// request a fresh capture per row, rename the dump dir to `name`).
inline const MVProbe* shotTable(int& count)
{
    static const MVProbe kShots[] = {
        {  0.25f,  0.0f, 0 }, { -0.25f, 0.0f, 0 },
        {  0.50f,  0.0f, 0 }, {  0.50f,  0.0f, 1 },
        { -0.50f,  0.0f, 1 }, {  1.00f,  0.0f, 1 },
        {  0.50f,  0.0f, 2 }, {  1.00f,  0.5f, 2 },
    };
    count = 8;
    return kShots;
}

} // namespace probe
