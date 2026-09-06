import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
dev="cuda:0"
m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
with torch.no_grad():
    for n,p in m.named_parameters():
        if "expands" in n and n.endswith("expand.weight"):
            p.zero_()
def forward_np(x_np):
    Hp, Wp = x_np.shape[:2]
    ph, pw = (-Hp)%16, (-Wp)%16
    c = F.pad(torch.from_numpy(x_np.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)
    d_ = torch.full((1,1,c.shape[2],c.shape[3]), 0.5, device=dev)
    mm = torch.zeros_like(c[:,:2])
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        o = m(c, d_, mm, c)
    return o[0,:,:Hp,:Wp].float().cpu().numpy().transpose(1,2,0)
i = 2
bef = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
off = ES.load_fp16_rgb(f"cap3_live/probes/comb/model_raw_{i:02d}.raw")
def score(rep, off, bef):
    return np.mean([np.corrcoef(rep[...,ch].ravel(), (off-bef)[...,ch].ravel())[0,1] for ch in range(3)])
rep_v = forward_np(bef)
print(f"as-captured        : corr={score(rep_v, off, bef):+.4f}")
befT = bef.transpose(1,0,2).copy()
offT = off.transpose(1,0,2).copy()
repT = forward_np(befT).transpose(1,0,2)
print(f"transposed         : corr={score(repT, off, bef):+.4f}")
print(f"transposed vs offT : corr={score(repT, offT, befT):+.4f}")
# also horizontal-comb frame f3 for both orientations
i = 3
bef = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
off = ES.load_fp16_rgb(f"cap3_live/probes/comb/model_raw_{i:02d}.raw")
rep = forward_np(bef)
print(f"f3 as-captured     : corr={score(rep, off, bef):+.4f}")
befT = bef.transpose(1,0,2).copy()
offT = off.transpose(1,0,2).copy()
repT = forward_np(befT).transpose(1,0,2)
print(f"f3 transposed      : corr={score(repT, off, bef):+.4f}")
