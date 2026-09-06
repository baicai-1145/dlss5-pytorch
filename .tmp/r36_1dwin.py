import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
import dlss5.swin_block as SB
dev="cuda:0"
m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
with torch.no_grad():
    for n,p in m.named_parameters():
        if "expands" in n and n.endswith("expand.weight"):
            p.zero_()
MODE = os.environ.get("WIN1D","h")   # h = 1x8 horizontal windows, v = 8x1
def wp_1d(x, ws):
    B,C,H,W = x.shape
    if MODE == "h":   # windows = 1 row x ws cols
        x = x.view(B, C, H, W//ws, ws)
        x = x.permute(0,2,3,4,1).contiguous()
        return x.view(-1, ws, C)
    else:             # windows = ws rows x 1 col
        x = x.view(B, C, H//ws, ws, W)
        x = x.permute(0,2,4,3,1).contiguous()
        return x.view(-1, ws, C)
def wr_1d(windows, ws, H, W, C):
    B = windows.shape[0]
    if MODE == "h":
        nH, nW = H, W//ws
        x = windows.view(B, nH, nW, ws, C)
        x = x.permute(0,4,1,2,3).contiguous()
        return x.view(B, C, H, W)
    else:
        nH, nW = H//ws, W
        x = windows.view(B, nH, nW, ws, C)
        x = x.permute(0,4,1,3,2).contiguous()
        return x.view(B, C, H, W)
# monkeypatch swin block partition/reverse + attention input shape
SB.window_partition = wp_1d
SB.window_reverse = wr_1d
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
for i in [2,3]:
    b = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
    o = ES.load_fp16_rgb(f"cap3_live/probes/comb/model_raw_{i:02d}.raw")
    r = forward_np(b)
    print(f"WIN1D={MODE} comb f{i}: corr={score(r, o, b):+.4f}", flush=True)
