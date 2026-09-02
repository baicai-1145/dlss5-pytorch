"""Phase 4 final: 3-zone decode (E4M3 | MX interleave | fp16 tail) + forward."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../dlss5'))
import numpy as np, torch
from dlss5.weights_loader import parse_records
from dlss5.calib_model import DLSS5NetCalib

BLOB = os.path.join(os.path.dirname(__file__), '..', '..', 'weights_blob.bin')

def e4m3_np(b):
    b = b.astype(np.int32)
    sgn = np.where((b & 0x80) != 0, -1.0, 1.0)
    exp = (b >> 3) & 0xF; man = b & 0x7
    val = np.where(exp == 0, man.astype(np.float64) * (2.0 ** -9),
                   (1.0 + man / 8.0) * np.power(2.0, exp.astype(np.float64) - 7))
    val = np.where(exp == 15, 0.0, val)
    return (sgn * val).astype(np.float32)

def fp16_np(b):
    u = np.frombuffer(b, dtype='<u2').astype(np.int32)
    s = np.where((u >> 15) != 0, -1.0, 1.0)
    e = (u >> 10) & 0x1F; m = u & 0x3FF
    v = np.where(e == 0, m.astype(np.float64) * (2.0 ** -24),
                 (1.0 + m / 1024.0) * np.power(2.0, e.astype(np.float64) - 15))
    v = np.where(e == 31, 0.0, v)
    return (s * v).astype(np.float32)

def classify_window(w):
    """512B window -> 'E' E4M3 | 'M' MX (w,s) | 'F' fp16 | 'Z' zero."""
    b = np.frombuffer(w, dtype=np.uint8)
    if not np.any(b):
        return 'Z'
    odd = b[1::2]
    u_o = len(set(odd.tolist()))
    top_frac = np.bincount(odd).max() / len(odd)
    med_o = float(np.median(odd))
    ve = e4m3_np(b)
    std_e4 = float(ve.std())
    u2 = np.frombuffer(w, dtype='<u2')
    sgn = np.where((u2 >> 15) != 0, -1.0, 1.0)
    e2 = ((u2 >> 10) & 0x1F).astype(np.int32); m2 = u2 & 0x3FF
    vf = np.where(e2 == 0, m2.astype(np.float64) * (2.0 ** -24),
                  (1.0 + m2 / 1024.0) * np.power(2.0, e2.astype(np.float64) - 15))
    nanf = (e2 == 31).mean()
    std_fp = float(np.where(e2 == 31, 0.0, vf).std())
    if std_e4 < 1.0:
        return 'E'
    if u_o <= 25 and top_frac >= 0.15 and 176 <= med_o <= 220:
        return 'M'
    if nanf < 0.01 and std_fp < 50.0:
        return 'F'
    return 'E'  # conservative fallback

def decode_record(seg: bytes):
    """Zone-decode one record; returns 1-D values (variable-length)."""
    vals = []
    n = len(seg)
    pos = 0
    while pos < n:
        end = min(pos + 512, n)
        w = seg[pos:end]
        if end - pos < 512:
            sub = np.frombuffer(w, dtype=np.uint8)
            odd = sub[1::2]
            u_o = len(set(odd.tolist())) if len(odd) else 99
            topf = np.bincount(odd).max() / len(odd) if len(odd) else 0
            med = float(np.median(odd)) if len(odd) else 0
            ve = e4m3_np(sub)
            if not np.any(sub): cls = 'Z'
            elif float(ve.std()) < 1.0: cls = 'E'
            elif u_o <= 25 and topf >= 0.15 and 176 <= med <= 220: cls = 'M'
            else: cls = 'F'
        else:
            cls = classify_window(w)
        arr = np.frombuffer(w, dtype=np.uint8)
        if cls == 'Z':
            pass
        elif cls == 'E':
            vals.append(e4m3_np(arr))
        elif cls == 'M':
            even = arr[0::2]; oddv = arr[1::2].astype(np.float64)
            wv = e4m3_np(even) * (2.0 ** (oddv - 205.0))
            vals.append(wv)
        elif cls == 'F':
            vals.append(fp16_np(w))
        pos = end
    return np.concatenate(vals) if vals else np.zeros(0, dtype=np.float32)

def main():
    recs = parse_records(BLOB)
    zone_counts = {'E': 0, 'M': 0, 'F': 0, 'Z': 0}
    streams = []
    with open(BLOB, 'rb') as f:
        for r in recs:
            f.seek(r['w_off'])
            seg = f.read(r['B'])
            # quick zone tally at 512B
            for p in range(0, len(seg) - 511, 512):
                cls = classify_window(seg[p:p+512])
                zone_counts[cls] += 1
            streams.append(decode_record(seg))
    print('zone tally (512B win):', zone_counts, file=sys.stderr)
    stream = np.concatenate(streams)
    print(f'decoded stream len: {len(stream):,}  (model params 147,683,778)', file=sys.stderr)

    m = DLSS5NetCalib().cuda().eval()
    params = [p for p in m.parameters()]
    total = sum(p.numel() for p in params)
    t = torch.from_numpy(stream)
    if t.numel() > total:
        t = t[:total]
    elif t.numel() < total:
        t = torch.cat([t, torch.zeros(total - t.numel())])
    print(f'filling {t.numel():,} -> {total:,}', file=sys.stderr)
    with torch.no_grad():
        off = 0
        for p in params:
            n = p.numel()
            p.copy_(t[off:off + n].reshape(p.shape).to(p.dtype))
            off += n

    stats = {}
    def hook(nm):
        def h(mod, i, o):
            a = o.detach().float()
            stats[nm] = (a.abs().mean().item(), a.abs().max().item(),
                         torch.isnan(a).sum().item())
        return h
    m.stem.register_forward_hook(hook('stem'))
    for i, st in enumerate(m.enc): st.register_forward_hook(hook(f'enc{i}'))
    for i, blk in enumerate(m.bn): blk.register_forward_hook(hook(f'bn{i}'))
    m.bn_proj.register_forward_hook(hook('bn_proj'))
    for i in range(5): m.dec[i].register_forward_hook(hook(f'dec{i}'))
    m.tail.register_forward_hook(hook('tail'))
    with torch.no_grad():
        c  = torch.rand(1, 3, 96, 144, device='cuda')
        d  = torch.rand(1, 1, 96, 144, device='cuda')
        mv = torch.rand(1, 2, 96, 144, device='cuda')
        ct = torch.rand(1, 3, 96, 144, device='cuda')
        out = m(c, d, mv, ct)
    print(f'out {tuple(out.shape)} NaN={torch.isnan(out).sum().item()} '
          f'absmean={out.abs().mean().item():.4e}')
    print('stage activation absmean / absmax / NaN:')
    for k, (mn, mx, nn) in stats.items():
        flag = '' if (0.001 < mn < 1e4 and nn == 0) else '  <<< CHECK'
        print(f'  {k:9s} {mn:12.4e} {mx:12.4e} NaN={nn}{flag}')

if __name__ == '__main__':
    main()
