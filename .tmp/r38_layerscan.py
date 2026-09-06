"""R38 Line b: layer-wise T-sensitivity scan.
Idea: if a global transpose inside our model cancels a transpose applied
to the INPUT, then feeding transposed input and comparing intermediate
layer activations against the normal-input run tells us WHERE the two
paths diverge first. Divergence layer = where the transpose is (or
isn't) corrected.
Concretely: run frame x and frame x^T through the model with hooks on
every module; compare activation STATISTICS (mean/std) per layer. The
first layer where x^T activations look like the MIRROR (transposed) of
x activations reveals the axis-swap point. Because a pure transpose bug
would make act(x^T) == transpose(act(x)) BEFORE the buggy layer and
diverge AFTER it.
Also directly test per-layer weight transposition experiments:
  - stem.weight kh/kw swap (done R36: flips orientation, no fix)
  - ALL conv layers kh/kw swap (stem + tail + any depthwise)
  - stem weight IN-CHANNEL order swap (color/depth/mv lane order)
  - window partition permute variant
Score each on comb f2/f3 + clean f13 held-out corr.
"""
import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
dev="cuda:0"

def load_model_fresh():
    m = dlss5.load_model("weights_blob.bin", device=dev); m.eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()
    return m

def score_comb_and_clean(m):
    out = {}
    # comb f2/f3 zero-motion single-shot
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
    for i in (2,3):
        b = ES.load_fp16_rgb(f"cap3_live/probes/comb/before_{i:02d}.raw")
        o = ES.load_fp16_rgb(f"cap3_live/probes/comb/model_raw_{i:02d}.raw")
        out[f"comb_f{i}"] = score(forward_np(b), o, b)
    return out

def swap_stem(m, which):
    w = m.stem.weight.data.clone()
    if which == "khkw":
        w = w.transpose(-1,-2).contiguous()
    elif which == "hflip":
        w = w.flip(-1).contiguous()
    elif which == "vflip":
        w = w.flip(-2).contiguous()
    with torch.no_grad():
        m.stem.weight.copy_(w)

variants = ["none", "khkw", "hflip", "vflip"]
for v in variants:
    m = load_model_fresh()
    if v != "none":
        swap_stem(m, v)
    r = score_comb_and_clean(m)
    print(f"stem={v:6s}: " + "  ".join(f"{k}={val:+.4f}" for k,val in r.items()), flush=True)
