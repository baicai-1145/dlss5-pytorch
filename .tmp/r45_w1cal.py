"""R45: W1 scale calibration. W1 = window-domain raw-sum RMS (faithful
SASS decode, comb-symmetric). Clean collapsed at -0.0757 — scale
mismatch hypothesis. Grid scan: divisor = sum(x^2)*2^k, k in {-13..+1}
plus calibrated gamma/beta rescale variants.
  V1: y = x*rsqrt(max(sum_w(x2)*2^k, floor))*gamma        (gamma-only)
  V2: V1 + beta
Accept: comb symmetry kept + clean16 >= 0.39.
Also record comb f2/f3 for every k.
"""
import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
import dlss5.swin_block as SB
from eval_cap3 import H, W
dev="cuda:0"
FLOOR = 6.198883056640625e-05
CFG = {"k": -13, "use_beta": False, "gpow": 1.0}
_LNF = torch.nn.LayerNorm.forward
def ln_fwd(self, x):
    if CFG["k"] is None or x.dim() != 4 or not (x.shape[1] % self.ws_dummy == 0 and x.shape[2] % self.ws_dummy == 0):
        return _LNF(self, x)
    B, Hp, Wp, C = x.shape
    gh, gw = Hp // self.ws_dummy, Wp // self.ws_dummy
    xr = x.view(B, gh, self.ws_dummy, gw, self.ws_dummy, C)
    s = (xr*xr).sum(dim=(2,3,4,5), keepdim=True).clamp_min(FLOOR)
    y = xr * torch.rsqrt(s * (2.0 ** CFG["k"])) * self.weight.view(1,1,1,1,C) ** CFG["gpow"]
    if CFG["use_beta"] and self.bias is not None:
        y = y + self.bias
    return y.view(B, Hp, Wp, C)
torch.nn.LayerNorm.forward = ln_fwd
def build():
    m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()
    for name, mod in m.named_modules():
        if isinstance(mod, SB.SwinBlock):
            mod.norm1.ws_dummy = mod.ws; mod.norm2.ws_dummy = mod.ws
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.LayerNorm) and not hasattr(mod, "ws_dummy"):
            mod.ws_dummy = 8
    return m
m = build()
def clean16():
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
        ctl = c if prev is None else ((1-WB)*c + WB*F.pad(prev,(0,pw,0,ph),mode="replicate")).clamp(0,1)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            o = m(c, d_, mm, ctl)
        rep = o[0,:,:H,:W].float().cpu().numpy().transpose(1,2,0)
        off = ES.load_fp16_rgb(f"cap3_live/clean/model_raw_{i:02d}.raw")
        if i in (8,10,13,15):
            cs.append(np.mean([np.corrcoef(rep[...,ch].ravel(), (off-bef)[...,ch].ravel())[0,1] for ch in range(3)]))
        prev = torch.from_numpy(rep.transpose(2,0,1))[None].to(dev)
    return float(np.mean(cs))
def comb_f23():
    out=[]
    for i in (2,3):
        b = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
        o = ES.load_fp16_rgb(f"cap3_live/probes/comb/model_raw_{i:02d}.raw")
        Hp,Wp = b.shape[:2]; ph2,pw2=(-Hp)%16,(-Wp)%16
        c = F.pad(torch.from_numpy(b.transpose(2,0,1))[None].float(),(0,pw2,0,ph2),mode="replicate").to(dev)
        d_ = torch.full((1,1,c.shape[2],c.shape[3]), 0.5, device=dev)
        mm = torch.zeros_like(c[:,:2])
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            rep = m(c, d_, mm, c)[0,:,:Hp,:Wp].float().cpu().numpy().transpose(1,2,0)
        out.append(np.mean([np.corrcoef(rep[...,ch].ravel(), (o-b)[...,ch].ravel())[0,1] for ch in range(3)]))
    return out
for name, cfg in [
    ("k=-13 (N=8192)", {"k":-13}), ("k=-12", {"k":-12}), ("k=-11", {"k":-11}),
    ("k=-10", {"k":-10}), ("k=-9", {"k":-9}), ("k=-8", {"k":-8}), ("k=-7", {"k":-7}),
    ("k=-6", {"k":-6}), ("k=-5", {"k":-5}), ("k=-4", {"k":-4}), ("k=-3", {"k":-3}),
    ("k=-2", {"k":-2}), ("k=-1", {"k":-1}), ("k=0 (raw sum)", {"k":0}),
]:
    CFG.update(cfg)
    c16 = clean16(); f2, f3 = comb_f23()
    print(f"{name:16s}: clean16={c16:+.4f} f2={f2:+.4f} f3={f3:+.4f}", flush=True)
