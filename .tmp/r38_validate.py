import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
from eval_cap3 import H, W
dev="cuda:0"
ph,pw=(-H)%16,(-W)%16
su=torch.tensor([-0.14,1.12],device=dev).view(1,2,1,1)
WB=0.25
def run(vflip_stem):
    m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()
        if vflip_stem:
            m.stem.weight.copy_(m.stem.weight.flip(-2))
    cs=[]
    prev=None
    for i in range(16):
        bef = ES.load_fp16_rgb(f"cap3_live/clean/before_{i:02d}.raw")
        dep = dlss5.load_dxgi_depth(f"cap3_live/clean/depth_{i:02d}.raw")
        mot = dlss5.load_dxgi_motion(f"cap3_live/clean/motion_{i:02d}.raw")
        c = F.pad(torch.from_numpy(bef.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)
        d_ = F.pad(torch.from_numpy(dep)[None,None].float(),(0,pw,0,ph),mode="replicate").to(dev)
        mm = F.pad(torch.from_numpy(mot.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)*su
        if prev is None:
            ctl = c
        else:
            prev_p = F.pad(prev, (0,pw,0,ph), mode="replicate")
            ctl = ((1-WB)*c + WB*prev_p).clamp(0,1)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            o = m(c, d_, mm, ctl)
        rep = o[0,:,:H,:W].float().cpu().numpy().transpose(1,2,0)
        off = ES.load_fp16_rgb(f"cap3_live/clean/model_raw_{i:02d}.raw")
        if i in (8,10,13,15):
            cs.append(np.mean([np.corrcoef(rep[...,ch].ravel(), (off-bef)[...,ch].ravel())[0,1] for ch in range(3)]))
        prev = torch.from_numpy(rep.transpose(2,0,1))[None].to(dev)
    return np.mean(cs)
print(f"baseline   : clean held corr = {run(False):+.4f}", flush=True)
print(f"stem-vflip : clean held corr = {run(True):+.4f}", flush=True)
