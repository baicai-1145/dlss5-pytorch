"""R47: W1@dec.4 full acceptance + baseline comparison.
1) clean16 per-frame corr (all 16, vs baseline per-frame)
2) comb f2/f3
3) oracle 4 groups (flat DC table corr, impulse radial corr, grayramp
   transfer corr, reset trajectory) — replica-vs-official, baseline vs W1
4) sharpness (Laplacian flat/edge) direction check on clean f13
Pedigree: architecture REVERSED (SASS window-domain reduction at the
last stage before tail). No calibration.
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
MODE = {"w1": False}
_LNF = torch.nn.LayerNorm.forward
def ln_fwd(self, x):
    if MODE["w1"] and getattr(self, "group", None) == "dec.4" and x.dim() == 4:
        ws = getattr(self, "ws_dummy", 8)
        B, Hp, Wp, C = x.shape
        if Hp % ws == 0 and Wp % ws == 0:
            gh, gw = Hp // ws, Wp // ws
            xr = x.view(B, gh, ws, gw, ws, C)
            s = (xr*xr).sum(dim=(2,3,4,5), keepdim=True).clamp_min(FLOOR)
            y = xr * torch.rsqrt(s) * self.weight.view(1,1,1,1,C)
            return y.view(B, Hp, Wp, C)
    return _LNF(self, x)
torch.nn.LayerNorm.forward = ln_fwd
def build():
    m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()
    for name, mod in m.named_modules():
        if isinstance(mod, SB.SwinBlock) and name.startswith("dec.4"):
            mod.norm1.ws_dummy = getattr(mod, "ws", 8); mod.norm1.group = "dec.4"
            mod.norm2.ws_dummy = getattr(mod, "ws", 8); mod.norm2.group = "dec.4"
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.LayerNorm) and not hasattr(mod, "group"):
            mod.ws_dummy = 8; mod.group = "other"
    return m
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
def lap_sharp(img):
    g = img @ LUMA
    lp = g[1:-1,1:-1] - 0.25*(g[:-2,1:-1] + g[2:,1:-1] + g[1:-1,:-2] + g[1:-1,2:])
    gy, gx = np.gradient(g)
    grad = np.sqrt(gx**2 + gy**2)
    lo, hi = np.percentile(grad, [40, 60])
    flat = grad <= lo; edge = grad >= hi
    lpf = np.zeros_like(g); lpf[1:-1,1:-1] = lp
    return {"flat": float(lpf[flat].std()), "edge": float(lpf[edge].std())}
def clean16_perframe():
    su = torch.tensor([-0.14,1.12],device=dev).view(1,2,1,1)
    ph,pw=(-H)%16,(-W)%16
    WB=0.25; rows=[]; prev=None
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
        cs = [np.corrcoef(rep[...,ch].ravel(), (off-bef)[...,ch].ravel())[0,1] for ch in range(3)]
        rows.append((i, float(np.mean(cs)), rep))
        prev = torch.from_numpy(rep.transpose(2,0,1))[None].to(dev)
    return rows
def comb():
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
def oracle_groups():
    import importlib.util
    spec = importlib.util.spec_from_file_location("oo", ".tmp/oracle_official.py")
    oo = importlib.util.module_from_spec(spec)
    sys.modules["oo"] = oo
    try:
        spec.loader.exec_module(oo)
    except SystemExit:
        pass
    o = np.load(".tmp/oracle_official.npz")
    res = {}
    # flat DC
    frames = []
    prev = None
    for i in range(16):
        bef = oo.load_color(f"{oo.P}/flat/before_{i:02d}.raw")
        dep = dlss5.load_dxgi_depth(f"{oo.P}/flat/depth_{i:02d}.raw")
        mot = dlss5.load_dxgi_motion(f"{oo.P}/flat/motion_{i:02d}.raw")
        c = torch.from_numpy(bef.transpose(2,0,1))[None].float()
        d_ = torch.from_numpy(dep)[None,None].float()
        mm = torch.from_numpy(mot.transpose(2,0,1))[None].float()*0.02
        r = c if prev is None else torch.from_numpy(prev.transpose(2,0,1))[None].float()
        ph,pw=(-H)%16,(-W)%16
        pad = lambda t: F.pad(t,(0,pw,0,ph),mode="replicate").to(dev)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            res_ = m(pad(c), pad(d_), pad(mm), pad(r))
        delta = res_[:,:, :H,:W].squeeze(0).permute(1,2,0).float().cpu().numpy()
        prev = bef + delta
        frames.append((bef, delta))
    lv2dc = {}
    for bef, delta in frames:
        lv = round(float(bef.reshape(-1,3).mean(0)[0]), 4)
        lv2dc.setdefault(lv, []).append(delta.reshape(-1,3).mean(0))
    off_levels = o["flat_levels"]; off_dc = o["flat_dc"]
    rep_dc = []
    for lv in off_levels:
        lv = round(float(lv), 4)
        rep_dc.append(np.stack(lv2dc[lv]).mean(0) if lv in lv2dc else np.full(3, np.nan))
    rep_dc = np.stack(rep_dc)
    ok = ~np.isnan(rep_dc[:,0])
    res["flat_rgb_corr"] = float(np.corrcoef(rep_dc[ok].ravel(), off_dc[ok].ravel())[0,1])
    res["flat_maxdiff"] = float(np.abs(rep_dc[ok]-off_dc[ok]).max())
    # impulse radial
    frames = []
    prev = None
    for i in range(16):
        bef = oo.load_color(f"{oo.P}/impulse/before_{i:02d}.raw")
        dep = dlss5.load_dxgi_depth(f"{oo.P}/impulse/depth_{i:02d}.raw")
        mot = dlss5.load_dxgi_motion(f"{oo.P}/impulse/motion_{i:02d}.raw")
        c = torch.from_numpy(bef.transpose(2,0,1))[None].float()
        d_ = torch.from_numpy(dep)[None,None].float()
        mm = torch.from_numpy(mot.transpose(2,0,1))[None].float()*0.02
        r = c if prev is None else torch.from_numpy(prev.transpose(2,0,1))[None].float()
        ph,pw=(-H)%16,(-W)%16
        pad = lambda t: F.pad(t,(0,pw,0,ph),mode="replicate").to(dev)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            res_ = m(pad(c), pad(d_), pad(mm), pad(r))
        delta = res_[:,:, :H,:W].squeeze(0).permute(1,2,0).float().cpu().numpy()
        prev = bef + delta
        frames.append((bef, delta))
    deltas = sum(delta for _, delta in frames)/len(frames)
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt((yy-835)**2 + (xx-1683)**2).astype(np.float32)
    net = deltas.sum(-1)
    prof_rep = []
    for bc, ov in zip(o["imp_radial_r"], o["imp_radial_net"]):
        msk = (rr >= bc-1) & (rr < bc+1)
        prof_rep.append(net[msk].mean() if msk.sum() else np.nan)
    prof_rep = np.array(prof_rep)
    ok = ~np.isnan(prof_rep)
    res["impulse_radial_corr"] = float(np.corrcoef(prof_rep[ok], o["imp_radial_net"][ok])[0,1])
    # grayramp transfer
    frames = []
    prev = None
    for i in range(16):
        bef = oo.load_color(f"{oo.P}/grayramp/before_{i:02d}.raw")
        dep = dlss5.load_dxgi_depth(f"{oo.P}/grayramp/depth_{i:02d}.raw")
        mot = dlss5.load_dxgi_motion(f"{oo.P}/grayramp/motion_{i:02d}.raw")
        c = torch.from_numpy(bef.transpose(2,0,1))[None].float()
        d_ = torch.from_numpy(dep)[None,None].float()
        mm = torch.from_numpy(mot.transpose(2,0,1))[None].float()*0.02
        r = c if prev is None else torch.from_numpy(prev.transpose(2,0,1))[None].float()
        ph,pw=(-H)%16,(-W)%16
        pad = lambda t: F.pad(t,(0,pw,0,ph),mode="replicate").to(dev)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            res_ = m(pad(c), pad(d_), pad(mm), pad(r))
        delta = res_[:,:, :H,:W].squeeze(0).permute(1,2,0).float().cpu().numpy()
        prev = bef + delta
        frames.append((bef, delta))
    raw_acc = sum(bef + delta for bef, delta in frames)/len(frames)
    col_rep = raw_acc.mean(0)
    luma_rep = col_rep @ LUMA; luma_off = o["ramp_col_raw"] @ LUMA; luma_bef = o["ramp_col_bef"] @ LUMA
    res["ramp_all_corr"] = float(np.corrcoef(luma_rep, luma_off)[0,1])
    res["ramp_delta_corr"] = float(np.corrcoef(luma_rep-luma_bef, luma_off-luma_bef)[0,1])
    # reset trajectory
    frames = []
    prev = None
    lumas = []
    dcs = []
    for i in range(16):
        bef = oo.load_color(f"{oo.P}/reset/before_{i:02d}.raw")
        dep = dlss5.load_dxgi_depth(f"{oo.P}/reset/depth_{i:02d}.raw")
        mot = dlss5.load_dxgi_motion(f"{oo.P}/reset/motion_{i:02d}.raw")
        c = torch.from_numpy(bef.transpose(2,0,1))[None].float()
        d_ = torch.from_numpy(dep)[None,None].float()
        mm = torch.from_numpy(mot.transpose(2,0,1))[None].float()*0.02
        r = c if prev is None else torch.from_numpy(prev.transpose(2,0,1))[None].float()
        ph,pw=(-H)%16,(-W)%16
        pad = lambda t: F.pad(t,(0,pw,0,ph),mode="replicate").to(dev)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            res_ = m(pad(c), pad(d_), pad(mm), pad(r))
        delta = res_[:,:, :H,:W].squeeze(0).permute(1,2,0).float().cpu().numpy()
        prev = bef + delta
        lumas.append(float((bef+delta).reshape(-1,3).mean(0) @ LUMA))
        dcs.append(delta.reshape(-1,3).mean(0))
    res["reset_luma_corr"] = float(np.corrcoef(lumas, o["reset_luma"])[0,1])
    res["reset_dc_corr"] = float(np.corrcoef(np.stack(dcs).ravel(), o["reset_dc"].ravel())[0,1])
    return res
results = {}
for label, w1 in [("baseline", False), ("W1@dec.4", True)]:
    MODE["w1"] = w1
    m = build()
    rows = clean16_perframe()
    f2, f3 = comb()
    og = oracle_groups()
    sh = lap_sharp(rows[13][2])
    print(f"=== {label} ===", flush=True)
    print("per-frame corr:", " ".join(f"{c:+.4f}" for _, c, _ in rows), flush=True)
    print(f"mean16={np.mean([c for _,c,_ in rows]):+.4f}  comb f2={f2:+.4f} f3={f3:+.4f}", flush=True)
    print(f"flat_rgb_corr={og['flat_rgb_corr']:+.4f} flat_maxdiff={og['flat_maxdiff']:.5f}", flush=True)
    print(f"impulse_radial_corr={og['impulse_radial_corr']:+.4f}", flush=True)
    print(f"ramp_all_corr={og['ramp_all_corr']:+.4f} ramp_delta_corr={og['ramp_delta_corr']:+.4f}", flush=True)
    print(f"reset_luma_corr={og['reset_luma_corr']:+.4f} reset_dc_corr={og['reset_dc_corr']:+.4f}", flush=True)
    print(f"sharp f13: flat={sh['flat']:.5f} edge={sh['edge']:.5f}", flush=True)
    results[label] = (rows, (f2,f3), og, sh)
np.savez(".tmp/r47_accept.npz",
         base_perframe=np.array([c for _,c,_ in results["baseline"][0]]),
         w1_perframe=np.array([c for _,c,_ in results["W1@dec.4"][0]]))
