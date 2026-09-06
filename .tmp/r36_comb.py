import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
from eval_cap3 import H, W
dev="cuda:0"
ph,pw=(-H)%16,(-W)%16
m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
with torch.no_grad():
    for n,p in m.named_parameters():
        if "expands" in n and n.endswith("expand.weight"):
            p.zero_()
def psnr(a, b):
    mse = np.mean((a.astype(np.float64)-b)**2)
    return 99.0 if mse < 1e-12 else 10*np.log10(1.0/mse)
results = []
prev = None
for i in range(16):
    bef = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
    off = ES.load_fp16_rgb(f"cap3_live/probes/comb/model_raw_{i:02d}.raw")
    dep_p = f"cap3_live/probes/comb/depth_{i:02d}.raw"
    mot_p = f"cap3_live/probes/comb/motion_{i:02d}.raw"
    dep = dlss5.load_dxgi_depth(dep_p) if os.path.exists(dep_p) else np.full((H,W), 0.5, np.float32)
    mot = dlss5.load_dxgi_motion(mot_p) if os.path.exists(mot_p) else np.zeros((H,W,2), np.float32)
    c = F.pad(torch.from_numpy(bef.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)
    d_ = F.pad(torch.from_numpy(dep)[None,None].float(),(0,pw,0,ph),mode="replicate").to(dev)
    mm = F.pad(torch.from_numpy(mot.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)*torch.tensor([-0.14,1.12],device=dev).view(1,2,1,1)
    if prev is None:
        ctl = c
    else:
        prev_p = F.pad(prev,(0,pw,0,ph),mode="replicate")
        ctl = ((1-0.25)*c + 0.25*prev_p).clamp(0,1)
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        o = m(c, d_, mm, ctl)
    rep = o[0,:,:H,:W].float().cpu().numpy().transpose(1,2,0)
    rep_full = np.clip(bef + rep, 0, 1)
    off_full = np.clip(off, 0, 1)
    cs = [np.corrcoef(rep[...,ch].ravel(), (off-bef)[...,ch].ravel())[0,1] for ch in range(3)]
    ps_rep = psnr(rep_full, off_full)
    ps_in  = psnr(bef, off_full)
    results.append((i, np.mean(cs), ps_rep, ps_in))
    prev = torch.from_numpy(rep.transpose(2,0,1))[None].to(dev)
    print(f"f{i:02d}: delta-corr={np.mean(cs):+.4f}  PSNR(rep|off)={ps_rep:.2f}dB  PSNR(in|off)={ps_in:.2f}dB  diff={ps_rep-ps_in:+.2f}", flush=True)
arr = np.array([r[1:] for r in results])
print(f"MEAN: delta-corr={arr[:,0].mean():+.4f}  PSNR(rep)={arr[:,1].mean():.2f}  PSNR(in)={arr[:,2].mean():.2f}  gap={arr[:,1].mean()-arr[:,2].mean():+.2f}dB")
