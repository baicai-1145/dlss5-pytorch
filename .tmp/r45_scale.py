"""R45 final: mixed-domain hypothesis. W1 (window raw-RMS) only on ONE
stage group, LN elsewhere. Groups: enc01, enc234, bn, dec, expands."""
import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
import dlss5.swin_block as SB
from eval_cap3 import H, W
dev="cuda:0"
FLOOR = 6.198883056640625e-05
SEL = {"groups": (), "scale": 1.0}
_LNF = torch.nn.LayerNorm.forward
def ln_fwd(self, x):
    if x.dim() != 4 or getattr(self, "group", None) not in SEL["groups"]:
        return _LNF(self, x)
    if not (x.shape[1] % self.ws_dummy == 0 and x.shape[2] % self.ws_dummy == 0):
        return _LNF(self, x)
    B, Hp, Wp, C = x.shape
    gh, gw = Hp // self.ws_dummy, Wp // self.ws_dummy
    xr = x.view(B, gh, self.ws_dummy, gw, self.ws_dummy, C)
    s = (xr*xr).sum(dim=(2,3,4,5), keepdim=True).clamp_min(FLOOR)
    y = xr * torch.rsqrt(s) * self.weight.view(1,1,1,1,C) * SEL["scale"]
    return y.view(B, Hp, Wp, C)
torch.nn.LayerNorm.forward = ln_fwd
def group_of(prefix):
    if prefix.startswith("enc.0") or prefix.startswith("enc.1"): return "enc01"
    if prefix.startswith(("enc.2","enc.3","enc.4")): return "enc234"
    if ".bn." in prefix or prefix.startswith("bn."): return "bn"
    if prefix.startswith("dec"): return "dec"
    if prefix.startswith("expands"): return "expands"
    return "other"
def build():
    m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()
    for name, mod in m.named_modules():
        if isinstance(mod, SB.SwinBlock):
            g = group_of(name)
            mod.norm1.ws_dummy = mod.ws; mod.norm1.group = g
            mod.norm2.ws_dummy = mod.ws; mod.norm2.group = g
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.LayerNorm) and not hasattr(mod, "group"):
            mod.ws_dummy = 8; mod.group = group_of(name)
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
ALL_GROUPS = ("enc01","enc234","bn","dec","expands")
for j in [-2, 0, 2, 4, 6, 8, 10, 12, 14]:
    SEL["groups"] = ALL_GROUPS; SEL["scale"] = float(2 ** j)
    c16 = clean16(); f2, f3 = comb_f23()
    print(f"W1all x2^{j:<+3d}      : clean16={c16:+.4f} f2={f2:+.4f} f3={f3:+.4f}", flush=True)
