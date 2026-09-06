import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
from eval_cap3 import H, W
dev="cuda:0"
def fresh():
    m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()
    return m
def vflip_conv(mod):
    with torch.no_grad():
        mod.weight.copy_(mod.weight.flip(-2))
def score_fn(m):
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
    r = {}
    for i in (2,3):
        b = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
        o = ES.load_fp16_rgb(f"cap3_live/probes/comb/model_raw_{i:02d}.raw")
        r[f"f{i}"] = score(forward_np(b), o, b)
    # clean f13 held-out corr (single frame, zero-history)
    b = ES.load_fp16_rgb("cap3_live/clean/before_13.raw")
    o = ES.load_fp16_rgb("cap3_live/clean/model_raw_13.raw")
    dep = dlss5.load_dxgi_depth("cap3_live/clean/depth_13.raw")
    mot = dlss5.load_dxgi_motion("cap3_live/clean/motion_13.raw")
    su = torch.tensor([-0.14,1.12],device=dev).view(1,2,1,1)
    ph, pw = (-H)%16, (-W)%16
    c = F.pad(torch.from_numpy(b.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)
    d_ = F.pad(torch.from_numpy(dep)[None,None].float(),(0,pw,0,ph),mode="replicate").to(dev)
    mm = F.pad(torch.from_numpy(mot.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)*su
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        out = m(c, d_, mm, c)
    rep = out[0,:,:H,:W].float().cpu().numpy().transpose(1,2,0)
    r["clean13"] = score(rep, o, b)
    return r
# variants: stem-vflip alone; stem+tail vflip; tail alone
m = fresh(); vflip_conv(m.stem)
print("stem-vflip        :", {k: round(v,4) for k,v in score_fn(m).items()}, flush=True)
m = fresh(); vflip_conv(m.stem); vflip_conv(m.tail.conv)
print("stem+tail-vflip   :", {k: round(v,4) for k,v in score_fn(m).items()}, flush=True)
m = fresh(); vflip_conv(m.tail.conv)
print("tail-vflip only   :", {k: round(v,4) for k,v in score_fn(m).items()}, flush=True)
