"""R42 line-a: DLSS4.5 exact RMS form ablation.
y = x * rsqrt_approx(sum(x^2)) * w   (no mean, no bias, no eps)
SASS ground truth for the DLL (cubin_01 5c40-6050): HFMA2 x*x
accumulation -> HADD2 pairwise -> SHFL.BFLY 0x2 cross-lane -> MUFU.RSQ
-> HMUL2 x*rsq (x99 = V multicast) — NO mean subtract, NO eps add (the
6.19888e-05 constant is a HMNMX2 floor = eps_FLOOR on the SUM, i.e.
rsqrt(max(sum, 6.2e-5)), not an eps ADD; sqrt semantics: 1/sqrt(6.2e-5)
= 127 — acts as an approximate rsqrt clamp for tiny sums).
Ablations (flat oracle 3 groups + clean 16-frame corr):
  A base          : LN(g,b) current
  B rms-g         : x*rsqrt(sum_sq/... ) * gamma  — note SASS shows NO 1/N
                    scaling anywhere before RSQ (no 1/32..1/3072 consts
                    found in scan) => plain sum, not variance
  C rms-g-epsfloor: B + sum clamped to floor 6.19888e-05
  D plain         : x * w only
  E rms-nog       : rsqrt(sum) * x (kill the loaded gamma — tests
                    "gamma is RMS weight" hypothesis)
Flat oracle = the 3 oracle probes (uniform patches) measured at 3-decimals
as in earlier rounds; clean = 16-frame held-out corr (frames 8,10,13,15).
"""
import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
from eval_cap3 import H, W
dev="cuda:0"
MODE = {"m": "A"}
FLOOR = 6.198883056640625e-05
import dlss5.swin_block as SB
_orig_ln = torch.nn.functional.layer_norm  # not used directly? SwinBlock uses nn.LayerNorm modules
# Patch nn.LayerNorm forward globally is messy; instead wrap module functional via
# torch.nn.LayerNorm.forward
_LNF = torch.nn.LayerNorm.forward
def ln_fwd(self, x):
    m = MODE["m"]
    if m == "A":
        return _LNF(self, x)
    wsh = (None,) * (x.dim() - 1)
    if m in ("B", "C"):
        s = (x * x).sum(-1, keepdim=True)
        if m == "C":
            s = s.clamp_min(FLOOR)
        return x * torch.rsqrt(s) * self.weight[wsh]
    if m == "D":
        return x * self.weight[wsh]
    if m == "E":
        s = (x * x).sum(-1, keepdim=True)
        return x * torch.rsqrt(s)
    return _LNF(self, x)
torch.nn.LayerNorm.forward = ln_fwd

def build():
    m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()
    return m
m = build()
def forward_clean16():
    su = torch.tensor([-0.14,1.12],device=dev).view(1,2,1,1)
    ph,pw=(-H)%16,(-W)%16
    WB=0.25; cs=[]; prev=None
    for i in range(16):
        bef = ES.load_fp16_rgb(f"cap3_live/clean/before_{i:02d}.raw")
        dep = dlss5.load_dxgi_depth(f"cap3_live/clean/depth_{i:02d}.raw")
        mot = dlss5.load_dxgi_motion(f"cap3_live/clean/motion_{i:02d}.raw")
        c = F.pad(torch.from_numpy(bef.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)
        d_ = F.pad(torch.from_numpy(dep)[None,None].float(),(0,pw,0,ph),mode="replicate").to(dev)
        mm = F.pad(torch.from_numpy(mot.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)*su
        if prev is None:
            ctl = c
        else:
            ctl = ((1-WB)*c + WB*F.pad(prev,(0,pw,0,ph),mode="replicate")).clamp(0,1)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            o = m(c, d_, mm, ctl)
        rep = o[0,:,:H,:W].float().cpu().numpy().transpose(1,2,0)
        off = ES.load_fp16_rgb(f"cap3_live/clean/model_raw_{i:02d}.raw")
        if i in (8,10,13,15):
            cs.append(np.mean([np.corrcoef(rep[...,ch].ravel(), (off-bef)[...,ch].ravel())[0,1] for ch in range(3)]))
        prev = torch.from_numpy(rep.transpose(2,0,1))[None].to(dev)
    return float(np.mean(cs))
for name, mode in [("A LN(g,b) base","A"), ("B rms*gamma","B"), ("C rms*gamma+floor","C"), ("D x*w plain","D"), ("E rms no-gamma","E")]:
    MODE["m"] = mode
    c16 = forward_clean16()
    print(f"{name:18s}: clean16={c16:+.4f}", flush=True)
