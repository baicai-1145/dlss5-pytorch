"""Forward-verification diagnostics for Phase 4 semantic_fill.

Tasks A (bn chain sub-layer hooks), B (MX bias scan), C (dec0 fill integrity).

Runs once per process and prints a single report.  Reuses the same model/load
that semantic_fill.main() builds, so results are directly comparable.
"""
from __future__ import annotations
import os, sys, json, time, contextlib, io
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase3'))
from dlss5.calib_model import DLSS5NetCalib, _SplitBlock
import semantic_fill as sf


def capture_dec0_fill_stats(by_block):
    """Task C — pre-fill std vs post-fill std for dec.0.blocks.0 params.

    Returns dict of {param_name: (pre_std, post_std)} and a list of unchanged
    parameters (post==pre within atol) so we know which slots really got loaded.
    """
    m = DLSS5NetCalib().eval()
    pmap = dict(m.named_parameters())
    pre = {n: p.detach().clone() for n, p in pmap.items() if n.startswith('dec.0.blocks.0.')}

    unfilled, _ = sf.fill_model(m, by_block)
    same, diff = [], {}
    for n, pre_t in pre.items():
        post_t = pmap[n].detach()
        if torch.allclose(pre_t, post_t, atol=1e-8):
            same.append(n)
        else:
            diff[n] = (pre_t.std().item(), post_t.std().item(),
                       post_t.abs().mean().item())
    return same, diff


def hook_bn_sublayers(m, target_idx):
    """Per _SplitBlock sub-layer absmean trace.

    _SplitBlock.forward order (calib_model.py L97-L114):
        ffwd (512->2048)
        wqkv (2048->2048)
        proj (2048->2048)
        fold 2048->512 by mean-4
        side (512->6144)
        side fold 6144->512
        sum + permute back
    We hook the 4 Linear modules plus capture the 'fold' intermediate and final.
    """
    blk = m.bn[target_idx]
    st = {}

    def grab(name):
        def h(_, inp, out):
            a = out.detach().float()
            st[name] = {
                'absmean': a.abs().mean().item(),
                'absmax': a.abs().max().item(),
                'shape': tuple(a.shape),
            }
        return h

    hs = [
        blk.ffwd.register_forward_hook(grab('ffwd')),
        blk.wqkv.register_forward_hook(grab('wqkv')),
        blk.proj.register_forward_hook(grab('proj')),
        blk.side.register_forward_hook(grab('side')),
    ]
    return st, hs


def run_with_inner_hooks(m, idx):
    """Run one forward and capture per-sublayer stats for bn[idx]."""
    inner, hs = hook_bn_sublayers(m, idx)
    with torch.no_grad():
        out = m(torch.rand(1, 3, 64, 64), torch.rand(1, 1, 64, 64),
                torch.rand(1, 2, 64, 64), torch.rand(1, 3, 64, 64))
    for h in hs:
        h.remove()
    return inner, out.abs().mean().item()


def stage_hooks(m):
    s = {}
    def h(nm):
        def fn(mod, i, o):
            a = o.detach().float()
            s[nm] = (a.abs().mean().item(), a.abs().max().item())
        return fn
    h_list = [m.stem.register_forward_hook(h('stem'))]
    for i, st in enumerate(m.enc):
        h_list.append(st.register_forward_hook(h(f'enc{i}')))
    for i, blk in enumerate(m.bn):
        h_list.append(blk.register_forward_hook(h(f'bn{i}')))
    h_list.append(m.bn_proj.register_forward_hook(h('bn_proj')))
    for i in range(5):
        h_list.append(m.dec[i].register_forward_hook(h(f'dec{i}')))
    h_list.append(m.tail.register_forward_hook(h('tail')))
    return s, h_list


def run_with_stage_hooks(m, bias=None):
    """Run fwd at the requested bias override.

    If bias is None, uses the loaded values verbatim.  Otherwise, the by_block
    was already decoded with the requested bias — caller responsibility.
    """
    s, hs = stage_hooks(m)
    with torch.no_grad():
        out = m(torch.rand(1, 3, 64, 64), torch.rand(1, 1, 64, 64),
                torch.rand(1, 2, 64, 64), torch.rand(1, 3, 64, 64))
    for h in hs:
        h.remove()
    return s, out.abs().mean().item()


def bias_sweep(biases):
    """Task B — for each bias, re-decode by_block, refill, run forward."""
    results = []
    blob = open(sf.BLOB, 'rb').read()
    for b in biases:
        by_block = {}
        for name, B, off in sf.walk_records(blob):
            bn = int(name.split('.')[0][5:])
            layer = name.split('.')[1]
            raw = blob[off + 4:off + B] if B > 4 else blob[off:off + B]
            arr = np.frombuffer(raw, dtype=np.uint8)
            main, misc = decode_with_bias(arr, B, b)
            by_block.setdefault(bn, {})[layer] = (main, misc)
        m = DLSS5NetCalib().eval()
        sf.fill_model(m, by_block)
        s, am = run_with_stage_hooks(m, bias=b)
        results.append((b, am, s))
    return results


def decode_with_bias(arr, total_B, bias):
    """Mirror decode_record_raw but override MX bias."""
    if len(arr) < 512:
        return sf.e4m3_decode(arr), sf.fp16_decode(arr)
    if total_B in sf.MX_BOUND:
        lo, hi = sf.MX_BOUND[total_B]
        main = np.concatenate([sf.e4m3_decode(arr[:lo]),
                               sf.mx_decode_pairs(arr[lo:hi].reshape(-1, 2), bias=bias),
                               sf.e4m3_decode(arr[hi:])])
        return main, np.zeros(0, dtype=np.float32)
    if total_B in sf.FP16_TAIL and sf.FP16_TAIL[total_B]:
        t0 = sf.FP16_TAIL[total_B]
        return sf.e4m3_decode(arr[:t0]), sf.fp16_decode(arr[t0:])
    # Generic fallback — re-implement with bias override.
    tags = sf.classify_windows(arr)
    runs = []
    for t in tags:
        if runs and runs[-1][0] == t:
            runs[-1][1] += 1
        else:
            runs.append([t, 1])
    main_parts, misc_parts = [], []
    pos = 0
    for tag, n in runs:
        seg = arr[pos:pos + n * 512]
        if tag == 'E':
            main_parts.append(sf.e4m3_decode(seg))
        elif tag == 'M':
            main_parts.append(sf.mx_decode_pairs(seg[:len(seg) // 2 * 2].reshape(-1, 2), bias=bias))
        else:
            misc_parts.append(sf.fp16_decode(seg))
        pos += n * 512
    if pos < len(arr):
        main_parts.append(sf.e4m3_decode(arr[pos:]))
    main = np.concatenate([p for p in main_parts if len(p)]) if main_parts else np.zeros(0)
    misc = np.concatenate([p for p in misc_parts if len(p)]) if misc_parts else np.zeros(0, dtype=np.float32)
    return main, misc


def main():
    t0 = time.time()

    # ---- Load once (bias=205) for tasks A & C ----
    print('[load] decoding by_block (bias=205) ...')
    blob = open(sf.BLOB, 'rb').read()
    by_block_205 = {}
    for name, B, off in sf.walk_records(blob):
        bn = int(name.split('.')[0][5:])
        layer = name.split('.')[1]
        raw = blob[off + 4:off + B] if B > 4 else blob[off:off + B]
        arr = np.frombuffer(raw, dtype=np.uint8)
        main, misc = sf.decode_record_raw(arr, total_B=B)
        by_block_205.setdefault(bn, {})[layer] = (main, misc)
    print(f'[load] {len(by_block_205)} blocks  ({time.time()-t0:.1f}s)')

    # ---- Task C: dec0 fill integrity ----
    print('\n=== Task C: dec.0.blocks.0 fill integrity ===')
    m = DLSS5NetCalib().eval()
    pmap = dict(m.named_parameters())
    pre = {n: p.detach().clone() for n, p in pmap.items() if n.startswith('dec.0.blocks.0.')}
    pre_stds = {n: p.std().item() for n, p in pre.items()}
    sf.fill_model(m, by_block_205)
    same, diff = [], {}
    for n, pre_t in pre.items():
        post_t = pmap[n].detach()
        if torch.allclose(pre_t, post_t, atol=1e-8):
            same.append((n, pre_stds[n], post_t.std().item()))
        else:
            diff[n] = (pre_stds[n], post_t.std().item(), post_t.abs().mean().item())
    print(f'  unchanged after fill: {len(same)}/{len(pre)}')
    for n, ps, qs in same:
        print(f'    UNCHANGED std_pre={ps:.4f}  std_post={qs:.4f}  {n}')
    print(f'  changed after fill: {len(diff)}/{len(pre)}')
    for n, (ps, qs, am) in diff.items():
        print(f'    CHANGED  std_pre={ps:.4f}  std_post={qs:.4f}  absmean={am:.4f}  {n}')
    taskC_verdict = ('BUG: not all dec.0.blocks.0 params were written'
                     if len(same) >= len(pre) - 1 else 'all dec.0.blocks.0 params were written')

    # ---- Task A: bn0 and bn7 sub-layer hooks ----
    print('\n=== Task A: bn0 and bn7 sub-layer trace ===')
    inner0, out0 = run_with_inner_hooks(m, 0)
    inner7, out7 = run_with_inner_hooks(m, 7)
    for label, inner, am in [('bn0', inner0, out0), ('bn7', inner7, out7)]:
        print(f'  {label} fwd absmean={am:.4f}')
        for k in ('ffwd', 'wqkv', 'proj', 'side'):
            v = inner.get(k, {})
            print(f'    {label}.{k:5s} shape={v.get("shape","?")}  '
                  f'absmean={v.get("absmean",float("nan")):.4f}  '
                  f'absmax={v.get("absmax",float("nan")):.4f}')

    # Also re-run full stage scan for the report (so numbers match Task B format)
    s, _ = run_with_stage_hooks(m)
    # ^^ redundant; keep print of bn0..bn7 line
    print('  bn line for record:')
    for i in range(8):
        print(f'    bn{i}  {s[f"bn{i}"][0]:12.4e}  {s[f"bn{i}"][1]:12.4e}')

    # ---- Task B: MX bias sweep ----
    print('\n=== Task B: MX bias sensitivity scan ===')
    sweep = bias_sweep([203, 204, 205, 206, 206.5, 207])
    print('  bias   enc0     enc1     enc2     enc3     enc4     bn7       dec0      dec1      tail')
    for b, am, st in sweep:
        row = [f'{b:5.1f}']
        for nm in ('enc0','enc1','enc2','enc3','enc4','bn7','dec0','dec1','tail'):
            row.append(f'{st[nm][0]:.3e}')
        print('  ' + '  '.join(row))
        print(f'        out_absmean = {am:.4f}')

    print(f'\n[timing] total {time.time()-t0:.1f}s')
    print(f'[verdict C] {taskC_verdict}')


if __name__ == '__main__':
    main()