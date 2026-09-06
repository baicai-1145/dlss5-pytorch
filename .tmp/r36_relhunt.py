import numpy as np, sys, torch
sys.path.insert(0,"."); sys.path.insert(0,".tmp")
from dlss5.loader import parse_blob
blob = open("weights_blob.bin","rb").read()
pb = parse_blob(blob)
S = 15
def templates(H):
    i, j = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    t = {"manh": (np.abs(i-7)+np.abs(j-7)).ravel(),
         "row":  np.abs(i-7).repeat(S).ravel(),
         "col":  np.tile(np.abs(j-7), S).ravel(),
         "cheb": np.maximum(np.abs(i-7), np.abs(j-7)).ravel(),
         "sq":   ((i-7)**2+(j-7)**2).ravel()}
    return {k: ((v-v.mean())/(v.std()+1e-9)).astype(np.float32) for k,v in t.items()}
TEMPLATES = templates(S)
TEMPLATES = None
def templates(H):
    i, j = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    t = {
        "manh": (np.abs(i-7)+np.abs(j-7)).ravel(),
        "row":  np.abs(i-7).repeat(S).ravel(),
        "col":  np.tile(np.abs(j-7), S).ravel(),
        "cheb": np.maximum(np.abs(i-7), np.abs(j-7)).ravel(),
        "sq":   ((i-7)**2+(j-7)**2).ravel(),
    }
    return {k: ((v-v.mean())/(v.std()+1e-9)).astype(np.float32) for k,v in t.items()}
# target stages: enc.2 blocks (b9-13), enc.3 (b15-21), enc.4 (b23-29),
# dec.0 (b40-47) — c128/c256/c512. Their record ids:
from dlss5.rel_templates import TEMPLATES
targets = {9:225*4,10:225*4,11:225*4,12:225*4,13:225*4,
           15:225*8,16:225*8,17:225*8,18:225*8,19:225*8,20:225*8,21:225*8,
           23:225*16,24:225*16,25:225*16,26:225*16,27:225*16,28:225*16,29:225*16,
           40:225*16,41:225*16,42:225*16,43:225*16,44:225*16,45:225*16,46:225*16,47:225*16}
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
hits = []
for rid, Hvals in targets.items():
    if rid not in pb: continue
    for lname,(main, misc) in pb[rid].items():
        # decode full record payload as fp16 stream (both main and misc)
        for tag, buf in (("main", main.tobytes()), ("misc", misc.tobytes())):
            if len(buf) < Hvals*2: continue
            arr = np.frombuffer(buf, dtype=np.float16 if False else np.uint16)
            # try every 2-byte offset, stride 2 (fp16 alignment), use torch for speed
            t = torch.frombuffer(buf, dtype=torch.uint16)
            n = Hvals
            cand = []
            for off in range(0, min(len(t)-n, 200000), 2):
                w = t[off:off+n].view(torch.float16).float()
                if not torch.isfinite(w).all(): continue
                ws_ = (w - w.mean())/(w.std()+1e-9)
                if w.std() < 1e-3 or w.std() > 30: continue
                heads = n // 225
                if heads * 225 != n or heads < 1 or heads > 16: continue
                W2 = ws_.to(dev).view(heads, 225)
                best_k, best_c, best_h = None, 0, 0
                for k, tv in TEMPLATES.items():
                    tv_t = torch.from_numpy(np.tile(tv, heads)).to(dev).view(heads, 225)
                    cs = (W2 * tv_t).mean(dim=1)
                    mh = int(cs.abs().argmax())
                    if abs(float(cs[mh])) > abs(best_c):
                        best_c, best_k, best_h = float(cs[mh]), k, mh
                if abs(best_c) > 0.55:
                    cand.append((off, best_k, best_c, best_h))
            if cand:
                hits.append((rid, lname, tag, cand[:5]))
                print(f"rec{rid} {lname} {tag}: {len(cand)} hits, first: {cand[:3]}", flush=True)
print("DONE", len(hits))
