"""Micro-test: same input -> LN vs window-RMS output diff at dec.2.norm1.
Also: does the SwinBlock actually USE nn.LayerNorm or a custom class?"""
import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import torch, torch.nn.functional as F
import dlss5
import dlss5.swin_block as SB
dev="cuda:0"
m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
blk = m.dec[2].blocks[0]
print("norm1 type:", type(blk.norm1), "| norm2 type:", type(blk.norm2))
x = (torch.randn(1, 264, 480, 128, device=dev).abs()+0.01)
with torch.no_grad():
    y_ln = blk.norm1(x)
    B, Hp, Wp, C = x.shape
    xr = x.view(1, 33, 8, 60, 8, C)
    s = (xr*xr).sum(dim=(2,3,4,5), keepdim=True).clamp_min(6.198883056640625e-05)
    y_w1 = (xr * torch.rsqrt(s) * blk.norm1.weight.view(1,1,1,1,C)).view(B,Hp,Wp,C)
d = (y_ln - y_w1).abs()
print("LN vs W1: max|d| =", d.max().item(), " mean|d| =", d.mean().item())
print("LN out std:", y_ln.std().item(), " W1 out std:", y_w1.std().item())
xc = x[:, :1, :1, :].expand_as(x).contiguous()
with torch.no_grad():
    yl = blk.norm1(xc)
xrc = xc.view(1, 33, 8, 60, 8, C)
sc = (xrc*xrc).sum(dim=(2,3,4,5), keepdim=True)
yw = (xrc * torch.rsqrt(sc) * blk.norm1.weight.view(1,1,1,1,C)).view(B,Hp,Wp,C)
print("const-input: max|LN-W1| =", (yl-yw).abs().max().item())
