"""R44: reduction-domain ablation. SASS: x2 accumulate in-thread (HADD2
channel merge) -> SHFL.BFLY +-1/+-2 (cross-thread = window positions).
Reduction domain = WINDOW (pixels x C), not per-token. Variants:
  A  LN per-token (baseline)
  W1 x*rsqrt(max(sum_{win}(x2),floor))*gamma          [faithful decode]
  W2 x*rsqrt(mean_{win}(x2))*gamma                    [mean-square win]
  W3 x*rsqrt(max(sum_{win}(x2),floor))*gamma+beta
  W4 window-LN (mu_w, var_w) *gamma+beta
  T1 x*rsqrt(max(sum_C(x2),floor))*gamma              [per-token raw sum+floor]
Score: clean16 + comb f2/f3.
"""
import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
import dlss5.swin_block as SB
from eval_cap3 import H, W
dev="cuda:0"
FLOOR = 6.198883056640625e-05
MODE = {"m":"A"}
_LNF = torch.nn.LayerNorm.forward
def ln_fwd(self, x):
    m = MODE["m"]
    if m == "A" or x.dim() != 4:
        return _LNF(self, x)
    B, Hp, Wp, C = x.shape
    wsh = (None,)*3
    if m in ("W1","W3","T1","W4"):
        pass
    if m in ("W1","W3") and (Hp % self.ws_dummy == 0 and Wp % self.ws_dummy == 0):
        gh, gw = Hp // self.ws_dummy, Wp // self.ws_dummy
        xr = x.view(B, gh, self.ws_dummy, gw, self.ws_dummy, C)
        s = (xr*xr).sum(dim=(2,3,4,5), keepdim=True).clamp_min(FLOOR)
        f = torch.rsqrt(s).expand_as(xr) if False else torch.rsqrt(s)
        y = xr * f * self.weight.view(1,1,1,1,C) if False else (xr * f) * self.weight.view(1,1,1,1,C)
        y = y.view(B,Hp,Wp,C)
        if m == "W3" and self.bias is not None:
            y = y + self.bias
        return y
    if m == "W2" and (Hp % self.ws_dummy == 0 and Wp % self.ws_dummy == 0):
        gh, gw = Hp // self.ws_dummy, Wp // self.ws_dummy
        xr = x.view(B, gh, self.ws_dummy, gw, self.ws_dummy, C)
        N = self.ws_dummy*self.ws_dummy*C
        s = (xr*xr).mean(dim=(2,3,4,5), keepdim=True)
        y = (xr * torch.rsqrt(s.clamp_min(1e-12))) * self.weight.view(1,1,1,1,C)
        return y.view(B,Hp,Wp,C)
    if m == "W4" and (Hp % self.ws_dummy == 0 and Wp % self.ws_dummy == 0):
        gh, gw = Hp // self.ws_dummy, Wp // self.ws_dummy
        xr = x.view(B, gh, self.ws_dummy, gw, self.ws_dummy, C)
        mu = xr.mean(dim=(2,3,4,5), keepdim=True)
        var = ((xr-mu)**2).mean(dim=(2,3,4,5), keepdim=True)
        y = ((xr-mu)*torch.rsqrt(var.clamp_min(1e-12))) * self.weight.view(1,1,1,1,C)
        if self.bias is not None:
            y = y + self.bias.view(1,1,1,1,C)
        return y.view(B,Hp,Wp,C)
    if m == "T1" or (m in ("W1","W3") and not (Hp % self.ws_dummy == 0 and Wp % self.ws_dummy == 0)):
        s = (x*x).sum(-1, keepdim=True).clamp_min(FLOOR)
        y = x * torch.rsqrt(s) * self.weight[wsh]
        return y + self.bias if (m=="W3" and self.bias is not None) else y
    if m == "W2" and not (Hp % self.ws_dummy == 0 and Wp % self.ws_dummy == 0):
        s = (x*x).mean(-1, keepdim=True).clamp_min(1e-12)
        return x * torch.rsqrt(s) * self.weight[wsh]
    if m == "W4" and not (Hp % self.ws_dummy == 0 and Wp % self.ws_dummy == 0):
        return _LNF(self, x)
    return _LNF(self, x)
torch.nn.LayerNorm.forward = ln_fwd
# tag each LayerNorm with its block's ws: find SwinBlock norms after build
def build():
    m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()
    for name, mod in m.named_modules():
        if isinstance(mod, SB.SwinBlock):
            mod.norm1.ws_dummy = mod.ws
            mod.norm2.ws_dummy = mod.ws
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.LayerNorm) and not hasattr(mod, "ws_dummy"):
            mod.ws_dummy = 8
    return m
m = build()
def forward_clean16():
    su = torch.tensor([-0.14,1.12],device=dev).view(1,2,1,1)
    ph,pw=(-H)%16,(-W)%16
    WB=0.25; cs=[]; prev=None
    for i in range(16):
        bef = ES.load_fp16_rgb(f"cap3_live/clean/before_{i:02d}.raw")
        dep = dlss5.load_dxgi_depth(f"cap3_live/clean/depth_{i:02d}.raw")
        mot = dlss5.load_dxgi_motion(f"cap3_live/clean/motion_{i:02d}.raw")
        c = F.pad(torch.from_numpy(bef.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)
        d_ = F.pad(torch.from_numpy(dep)[None,None].float(),(0,pw,0,ph),mode="replicate").to(dev)
        mm = F.pad(torch.from_numpy(mot.transpose(2,0,1))[None].float(),(0,pw,0,ph),mode="replicate").to(dev)*su
        ctl = c if prev is None else ((1-WB)*c + WB*F.pad(prev,(0,pw,0,ph),mode="replicate")).clamp(0,1)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            o = m(c, d_, mm, ctl)
        rep = o[0,:,:H,:W].float().cpu().numpy().transpose(1,2,0)
        off = ES.load_fp16_rgb(f"cap3_live/clean/model_raw_{i:02d}.raw")
        if i in (8,10,13,15):
            cs.append(np.mean([np.corrcoef(rep[...,ch].ravel(), (off-bef)[...,ch].ravel())[0,1] for ch in range(3)]))
        prev = torch.from_numpy(rep.transpose(2,0,1))[None].to(dev)
    return float(np.mean(cs))
def comb_f23():
    out=[]
    for i in (2,3):
        b = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
        o = ES.load_fp16_rgb(f"cap3_live/probes/comb/model_raw_{i:02d}.raw")
        Hp,Wp = b.shape[:2]; ph2,pw2=(-Hp)%16,(-Wp)%16
        c = F.pad(torch.from_numpy(b.transpose(2,0,1))[None].float(),(0,pw2,0,ph2),mode="replicate").to(dev)
        d_ = torch.full((1,1,c.shape[2],c.shape[3]), 0.5, device=dev)
        mm = torch.zeros_like(c[:,:2])
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            rep = m(c, d_, mm, c)[0,:,:Hp,:Wp].float().cpu().numpy().transpose(1,2,0)
        out.append(np.mean([np.corrcoef(rep[...,ch].ravel(), (o-b)[...,ch].ravel())[0,1] for ch in range(3)]))
    return out
for name, mode in [("A LN base","A"), ("W1 winRMS sum+fl","W1"), ("W2 winRMS mean","W2"), ("W3 W1+beta","W3"), ("W4 winLN","W4"), ("T1 tokRMS sum+fl","T1")]:
    MODE["m"] = mode
    c16 = forward_clean16()
    f2, f3 = comb_f23()
    print(f"{name:17s}: clean16={c16:+.4f} f2={f2:+.4f} f3={f3:+.4f}", flush=True)
