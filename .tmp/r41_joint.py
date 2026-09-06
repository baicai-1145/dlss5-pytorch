"""R41 line-b: stem input lane order x vflip joint search.
Axes:
  CC  : color<->control group swap (equal width 3C)
  CR  : color channel reverse (RGB->BGR)
  KR  : control channel reverse
  VF  : stem kernel vflip (R38 parity axis)
8 combos + baseline, scored on comb f2/f3 + clean13 (single-frame).
"""
import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
from eval_cap3 import H, W
dev="cuda:0"
CFG = {"cc": False, "cr": False, "kr": False, "vf": False}
ORIG_CAT = torch.cat
def fake_cat(ts, dim):
    if dim==1 and len(ts)==4 and ts[0].shape[1]==3:
        c, d, m, k = ts
        if CFG["cc"]: c, k = k, c
        if CFG["cr"]: c = c.flip(1)
        if CFG["kr"]: k = k.flip(1)
        return ORIG_CAT([c, d, m, k], 1)
    return ORIG_CAT(ts, dim)
def build():
    m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()
        if CFG["vf"]:
            m.stem.weight.copy_(m.stem.weight.flip(-2))
    return m
def forward_np(m, x_np):
    torch.cat = fake_cat
    try:
        Hp, Wp = x_np.shape[:2]
        ph, pw = (-Hp)%16, (-Wp)%16
        c = F.pad(torch.from_numpy(x_np.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)
        d_ = torch.full((1,1,c.shape[2],c.shape[3]), 0.5, device=dev)
        mm = torch.zeros_like(c[:,:2])
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            o = m(c, d_, mm, c)
    finally:
        torch.cat = ORIG_CAT
    return o[0,:,:Hp,:Wp].float().cpu().numpy().transpose(1,2,0)
def score(rep, off, bef):
    return np.mean([np.corrcoef(rep[...,ch].ravel(), (off-bef)[...,ch].ravel())[0,1] for ch in range(3)])
m = build()
pairs=[]
for i in (2,3):
    b = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
    o = ES.load_fp16_rgb(f"cap3_live/probes/comb/model_raw_{i:02d}.raw")
    pairs.append((b,o))
b13 = ES.load_fp16_rgb("cap3_live/clean/before_13.raw")
o13 = ES.load_fp16_rgb("cap3_live/clean/model_raw_13.raw")
dep13 = dlss5.load_dxgi_depth("cap3_live/clean/depth_13.raw")
mot13 = dlss5.load_dxgi_motion("cap3_live/clean/motion_13.raw")
su = torch.tensor([-0.14,1.12],device=dev).view(1,2,1,1)
ph,pw=(-H)%16,(-W)%16
c13 = F.pad(torch.from_numpy(b13.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)
d13 = F.pad(torch.from_numpy(dep13)[None,None].float(),(0,pw,0,ph),mode="replicate").to(dev)
m13 = F.pad(torch.from_numpy(mot13.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)*su
def clean13():
    torch.cat = fake_cat
    try:
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            out = m(c13, d13, m13, c13)
    finally:
        torch.cat = ORIG_CAT
    rep = out[0,:,:H,:W].float().cpu().numpy().transpose(1,2,0)
    return score(rep, o13, b13)
for name, cfg in [("base          ",{}), ("CC            ",{"cc":True}), ("CR            ",{"cr":True}),
                  ("KR            ",{"kr":True}), ("VF            ",{"vf":True}),
                  ("CC+VF         ",{"cc":True,"vf":True}), ("CR+VF         ",{"cr":True,"vf":True}),
                  ("KR+VF         ",{"kr":True,"vf":True}), ("CC+CR+KR+VF   ",{"cc":True,"cr":True,"kr":True,"vf":True})]:
    CFG.update(cc=False,cr=False,kr=False,vf=False); CFG.update(cfg)
    m = build()
    f2 = score(forward_np(m, pairs[0][0]), pairs[0][1], pairs[0][0])
    f3 = score(forward_np(m, pairs[1][0]), pairs[1][1], pairs[1][0])
    c13v = clean13()
    print(f"{name}: f2={f2:+.4f} f3={f3:+.4f} clean13={c13v:+.4f}", flush=True)
