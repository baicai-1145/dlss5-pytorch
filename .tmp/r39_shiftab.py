"""R39 step4: shift-axis A/B via roll override (patch torch.roll within
SwinBlock.forward scope by wrapping forward and intercepting shift sizes)."""
import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
import dlss5.swin_block as SB
dev="cuda:0"
STATE = {"sh": (4,4)}
_orig_roll = torch.roll
def fake_roll(x, shifts, dims):
    # only intercept the SwinBlock feature rolls (dims (1,2) on 4D channel-last)
    if x.dim()==4 and tuple(dims)==(1,2):
        s = STATE["sh"]
        return _orig_roll(x, shifts=(-s[0], -s[1]), dims=dims)
    return _orig_roll(x, shifts, dims)
SB_FWD = SB.SwinBlock.forward
def fwd(self, x):
    torch.roll = fake_roll
    try:
        # negative roll for back-shift too: fake_roll negates — so pass +size
        # and fake_roll applies -s... need sign-aware: original forward calls
        # roll(-s) then roll(+s). Intercept both: use magnitude only.
        out = SB_FWD(self, x)
    finally:
        torch.roll = _orig_roll
    return out
SB.SwinBlock.forward = fwd

def fresh():
    m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()
    return m
def forward_np(m, x_np):
    Hp, Wp = x_np.shape[:2]
    ph, pw = (-Hp)%16, (-Wp)%16
    c = F.pad(torch.from_numpy(x_np.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)
    d_ = torch.full((1,1,c.shape[2],c.shape[3]), 0.5, device=dev)
    mm = torch.zeros_like(c[:,:2])
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        o = m(c, d_, mm, c)
    return o[0,:,:Hp,:Wp].float().cpu().numpy().transpose(1,2,0)
def score(rep, off, bef):
    return np.mean([np.corrcoef(rep[...,ch].ravel(), (off-bef)[...,ch].ravel())[0,1] for ch in range(3)])
m = fresh()
comb = []
for i in (2,3):
    b = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
    o = ES.load_fp16_rgb(f"cap3_live/probes/comb/model_raw_{i:02d}.raw")
    comb.append((b,o))
b13 = ES.load_fp16_rgb("cap3_live/clean/before_13.raw")
o13 = ES.load_fp16_rgb("cap3_live/clean/model_raw_13.raw")
dep13 = dlss5.load_dxgi_depth("cap3_live/clean/depth_13.raw")
mot13 = dlss5.load_dxgi_motion("cap3_live/clean/motion_13.raw")
from eval_cap3 import H, W
su = torch.tensor([-0.14,1.12],device=dev).view(1,2,1,1)
ph,pw=(-H)%16,(-W)%16
c13 = F.pad(torch.from_numpy(b13.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)
d13 = F.pad(torch.from_numpy(dep13)[None,None].float(),(0,pw,0,ph),mode="replicate").to(dev)
m13 = F.pad(torch.from_numpy(mot13.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)*su
def clean13():
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        out = m(c13, d13, m13, c13)
    rep = out[0,:,:H,:W].float().cpu().numpy().transpose(1,2,0)
    return score(rep, o13, b13)
for name, s in [("base(-4,-4)",(4,4)), ("hOnly(-4,0)",(4,0)), ("vOnly(0,-4)",(0,4)), ("mag2(-2,-2)",(2,2)), ("mag6(-6,-6)",(6,6))]:
    STATE["sh"] = s
    f2 = score(forward_np(m, comb[0][0]), comb[0][1], comb[0][0])
    f3 = score(forward_np(m, comb[1][0]), comb[1][1], comb[1][0])
    c13v = clean13()
    print(f"{name:12s}: f2={f2:+.4f} f3={f3:+.4f} clean13={c13v:+.4f}", flush=True)
