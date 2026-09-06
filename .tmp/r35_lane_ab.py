import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'.tmp')
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
from eval_cap3 import H, W
dev='cuda:0'
ph,pw=(-H)%16,(-W)%16
def warp_bicubic(color, mvx, mvy):
    h, w = color.shape[-2:]
    ys, xs = torch.meshgrid(torch.linspace(-1,1,h,device=color.device),
                            torch.linspace(-1,1,w,device=color.device), indexing='ij')
    gx = (xs + mvx[0,0]/w*2).float(); gy = (ys + mvy[0,0]/h*2).float()
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    return F.grid_sample(color, grid, mode='bicubic', padding_mode='border', align_corners=False)
def run(mode):
    m = dlss5.load_model('weights_blob.bin', device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if 'expands' in n and n.endswith('expand.weight'):
                p.zero_()
    cs=[]
    for i in [8,10,13,15]:
        bef = ES.load_fp16_rgb(f'cap3_live/clean/before_{i:02d}.raw')
        dep = dlss5.load_dxgi_depth(f'cap3_live/clean/depth_{i:02d}.raw')
        mot = dlss5.load_dxgi_motion(f'cap3_live/clean/motion_{i:02d}.raw')
        c = F.pad(torch.from_numpy(bef.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode='replicate').to(dev)
        d_ = F.pad(torch.from_numpy(dep)[None,None].float(),(0,pw,0,ph),mode='replicate').to(dev)
        mm = F.pad(torch.from_numpy(mot.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode='replicate').to(dev)
        if mode=='warp1':
            wc = warp_bicubic(c, mm[:,0:1], mm[:,1:2]); ctl = wc
        elif mode=='warpR12':
            mm_px = mm * torch.tensor([-0.14,1.12],device=dev).view(1,2,1,1)
            ctl = warp_bicubic(c, mm_px[:,0:1], mm_px[:,1:2])
        elif mode=='zero':
            ctl = torch.zeros_like(c)
        elif mode=='gauss':
            g = torch.Generator(device='cpu'); g.manual_seed(0x9E3779B1)
            ctl = torch.randn(c.shape, generator=g).to(dev)*0.1
        else:
            ctl = c
        mm_s = mm * torch.tensor([-0.14,1.12],device=dev).view(1,2,1,1)
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            o = m(c, d_, mm_s, ctl)
        r = o[0,:,:H,:W].float().cpu().numpy()
        mr = ES.load_fp16_rgb(f'cap3_live/clean/model_raw_{i:02d}.raw')
        f = (mr-bef).transpose(2,0,1)
        cs.append(np.mean([np.corrcoef(r[ch].ravel(), f[ch].ravel())[0,1] for ch in range(3)]))
    return np.mean(cs)
for mode in ['base','warp1','warpR12','zero','gauss']:
    print(f"{mode:8s}: corr(held) = {run(mode):+.4f}", flush=True)
