"""R40 step 1/3: window-geometry A/B — variant windows (no roll) with
three offset conventions. Shifted blocks only; offset (o_h,o_w) in
{(0,0),(2,2),(4,4)}; mask + S2-style zeroing at the crossed boundary.
 comb f2/f3 single-frame as the discriminator.
"""
import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
import dlss5.swin_block as SB
from eval_cap3 import H, W
dev="cuda:0"
STATE = {"off": (4,4)}
ORIG_FWD = SB.SwinBlock.forward
def fwd(self, x):
    if not self.shift:
        return ORIG_FWD(self, x)
    B, C, Hx, Wx = x.shape
    ws = self.ws
    pad_h = (ws - Hx % ws) % ws; pad_w = (ws - Wx % ws) % ws
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    x = x.permute(0, 2, 3, 1)
    hp, wp = Hx + pad_h, Wx + pad_w
    shortcut = x
    xn = self.norm1(x)
    xn = self.actq(xn)
    oh, ow = STATE["off"]
    gh, gw = hp // ws, wp // ws
    wgrid = xn.view(B, gh, ws, gw, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B*gh*gw, ws*ws, C)
    # mask: build index grids BEFORE windowing
    hid = torch.arange(hp, device=x.device).view(1, hp, 1).expand(B, hp, wp)
    wid = torch.arange(wp, device=x.device).view(1, 1, wp).expand(B, hp, wp)
    qh = ((hid + oh) % hp) // ws  # query window id under offset convention
    kh_ = ((hid + 2*oh) % hp) // ws
    qw = ((wid + ow) % wp) // gw if False else ((wid + ow) % wp) // ws
    kw_ = ((wid + 2*ow) % wp) // ws
    mask = ((qh != kh_) | (qw != kw_)).float()
    mw = mask.view(B, gh, ws, gw, ws).permute(0, 1, 3, 2, 4).reshape(B*gh*gw, 1, ws*ws)
    attn_out = self.attn(wgrid, None)
    # apply mask post-hoc: need pre-weights; instead zero via mask multiply on
    # attention weights is inside attn... approximate: zero tokens whose
    # query crosses the boundary BEFORE attention (kills cross-window mixing)
    keep = (mw.squeeze(1) > 0)  # (Bwin, N) tokens on the "same window" side
    wgrid2 = wgrid.clone()
    wgrid2[~keep] = 0
    attn_out = self.attn(wgrid2, None)
    x = window_reverse_like(attn_out, ws, B, gh, gw, C)
    if self.shift_size > 0:
        pass
    x = shortcut + x
    nx = self.actq(self.norm2(x))
    h = self.mlp[0](nx); h = self.mlp[1](h); h = self.actq(h)
    x = x + self.mlp[2](h)
    if pad_h or pad_w:
        x = x[:, :Hx, :Wx, :]
    return x.permute(0, 3, 1, 2)
def window_reverse_like(windows, ws, B, gh, gw, C):
    x = windows.view(B, gh, gw, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, gh*ws, gw*ws, C)
    return x
SB.SwinBlock.forward = fwd
def fresh():
    m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()
    return m
m = fresh()
def forward_np(x_np):
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
pairs = []
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
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        out = m(c13, d13, m13, c13)
    rep = out[0,:,:H,:W].float().cpu().numpy().transpose(1,2,0)
    return score(rep, o13, b13)
for name, off in [("roll-base(skip)","skip"), ("off(0,0)",(0,0)), ("off(2,2)",(2,2)), ("off(4,4)",(4,4))]:
    if off == "skip":
        print(f"{name:14s}: reference values from R39 (f2=-0.31 f3=+0.98)", flush=True)
        continue
    STATE["off"] = off
    f2 = score(forward_np(pairs[0][0]), pairs[0][1], pairs[0][0])
    f3 = score(forward_np(pairs[1][0]), pairs[1][1], pairs[1][0])
    c13v = clean13()
    print(f"{name:14s}: f2={f2:+.4f} f3={f3:+.4f} clean13={c13v:+.4f}", flush=True)
