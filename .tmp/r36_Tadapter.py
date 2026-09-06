import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
from eval_cap3 import H, W
dev="cuda:0"
m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
with torch.no_grad():
    for n,p in m.named_parameters():
        if "expands" in n and n.endswith("expand.weight"):
            p.zero_()
def forward_T(x_np):
    xt = np.ascontiguousarray(x_np.transpose(1,0,2))
    Hp, Wp = xt.shape[:2]
    ph, pw = (-Hp)%16, (-Wp)%16
    c = F.pad(torch.from_numpy(xt.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)
    d_ = torch.full((1,1,c.shape[2],c.shape[3]), 0.5, device=dev)
    mm = torch.zeros_like(c[:,:2])
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        o = m(c, d_, mm, c)
    return o[0,:,:Hp,:Wp].float().cpu().numpy().transpose(1,2,0).transpose(1,0,2)
def score(rep, off, bef):
    return np.mean([np.corrcoef(rep[...,ch].ravel(), (off-bef)[...,ch].ravel())[0,1] for ch in range(3)])
for i in [2,3]:
    b = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
    o = ES.load_fp16_rgb(f"cap3_live/probes/comb/model_raw_{i:02d}.raw")
    rT = forward_T(b)
    print(f"comb f{i}: T-adapter corr={score(rT, o, b):+.4f}", flush=True)
# clean f13 with full native pipeline in transposed domain
ph, pw = (-H)%16, (-W)%16
su = torch.tensor([-0.14,1.12],device=dev).view(1,2,1,1)
i = 13
bef = ES.load_fp16_rgb(f"cap3_live/clean/before_{i:02d}.raw")
off = ES.load_fp16_rgb(f"cap3_live/clean/model_raw_{i:02d}.raw")
dep = dlss5.load_dxgi_depth(f"cap3_live/clean/depth_{i:02d}.raw")
mot = dlss5.load_dxgi_motion(f"cap3_live/clean/motion_{i:02d}.raw")
# transpose everything
bT = np.ascontiguousarray(bef.transpose(1,0,2))
oT = np.ascontiguousarray(off.transpose(1,0,2))
depT = np.ascontiguousarray(dep.T)
motT = np.ascontiguousarray(mot.transpose(1,0,2))
HT, WT = W, H  # swapped
phT, pwT = (-HT)%16, (-WT)%16
c = F.pad(torch.from_numpy(bT.transpose(2,0,1))[None].float(),(0,pwT,0,phT),mode="replicate").to(dev)
d_ = F.pad(torch.from_numpy(depT)[None,None].float(),(0,pwT,0,phT),mode="replicate").to(dev)
mm = F.pad(torch.from_numpy(motT.transpose(2,0,1))[None].float(),(0,pwT,0,phT),mode="replicate").to(dev)*su
with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
    o = m(c, d_, mm, c)
repT = o[0,:,:HT,:WT].float().cpu().numpy().transpose(1,2,0)  # in T-domain
rep_back = repT.transpose(1,0,2)  # back to original orientation
print(f"clean f13: T-adapter corr={score(rep_back, off, bef):+.4f}")
