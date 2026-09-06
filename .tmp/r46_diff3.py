"""Empirical: W1@dec.2-only vs baseline raw output diff (identical?).
Plus dec.0-only, dec.1-only isolation."""
import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
import dlss5.swin_block as SB
dev="cuda:0"
FLOOR = 6.198883056640625e-05
SEL = {"groups": ()}
_LNF = torch.nn.LayerNorm.forward
def ln_fwd(self, x):
    g = getattr(self, "group", None)
    if x.dim() != 4 or g not in SEL["groups"]:
        return _LNF(self, x)
    ws = getattr(self, "ws_dummy", 8)
    B, Hp, Wp, C = x.shape
    if Hp % ws or Wp % ws:
        return _LNF(self, x)
    gh, gw = Hp // ws, Wp // ws
    xr = x.view(B, gh, ws, gw, ws, C)
    s = (xr*xr).sum(dim=(2,3,4,5), keepdim=True).clamp_min(FLOOR)
    y = xr * torch.rsqrt(s) * self.weight.view(1,1,1,1,C)
    return y.view(B, Hp, Wp, C)
torch.nn.LayerNorm.forward = ln_fwd
def stage_of(prefix):
    import re
    m = re.match(r"(enc\.\d|dec\.\d|bn\.\d|expands\.\d)", prefix)
    return m.group(1) if m else prefix.split(".")[0]
def build():
    m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()
    for name, mod in m.named_modules():
        if isinstance(mod, SB.SwinBlock):
            g = stage_of(name)
            mod.norm1.ws_dummy = getattr(mod, "ws", 8); mod.norm1.group = g
            mod.norm2.ws_dummy = getattr(mod, "ws", 8); mod.norm2.group = g
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.LayerNorm) and not hasattr(mod, "group"):
            mod.ws_dummy = 8; mod.group = stage_of(name)
    return m
m = build()
def comb_raw(i):
    b = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
    Hp,Wp = b.shape[:2]; ph2,pw2=(-Hp)%16,(-Wp)%16
    c = F.pad(torch.from_numpy(b.transpose(2,0,1))[None].float(),(0,pw2,0,ph2),mode="replicate").to(dev)
    d_ = torch.full((1,1,c.shape[2],c.shape[3]), 0.5, device=dev)
    mm = torch.zeros_like(c[:,:2])
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        rep = m(c, d_, mm, c)[0,:,:Hp,:Wp].float().cpu().numpy().transpose(1,2,0)
    return rep
def score_raw(rep, i):
    o = ES.load_fp16_rgb(f"cap3_live/probes/comb/model_raw_{i:02d}.raw")
    b = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
    return np.mean([np.corrcoef(rep[...,ch].ravel(), (o-b)[...,ch].ravel())[0,1] for ch in range(3)])
SEL["groups"] = ()
base2 = comb_raw(2)
for name, groups in [("dec.3",("dec.3",)), ("dec.4",("dec.4",)), ("dec.2+3",("dec.2","dec.3")), ("dec.3+4",("dec.3","dec.4"))]:
    SEL["groups"] = groups
    w2 = comb_raw(2)
    d = np.abs(base2 - w2).max()
    print(f"W1@{name:8s}: max|Δraw|={d:.6f}  f2={score_raw(w2,2):+.4f}", flush=True)
SEL["groups"] = ()
