"""Phase 4 fwdtest diagnostic — median-8 (HEAD) vs fixed-bias sweep + seeded fwd + bn sublayer + rel_bias impact.

Read-only: does NOT modify phase4/semantic_fill.py. Patches mx_decode_pairs via monkey-patch.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase3'))
sys.path.insert(0, HERE)
from dlss5.calib_model import DLSS5NetCalib, _SplitBlock
import semantic_fill as sf


# ---------- seeded helpers ----------
SEED_MODEL = 42
SEED_INPUT = 7


def stage_hooks(m):
    """Attach absmean/absmax/NaN hooks to all top-level stages."""
    s = {}
    def make(nm):
        def h(mod, i, o):
            a = o.detach().float()
            s[nm] = (a.abs().mean().item(), a.abs().max().item(),
                     int(torch.isnan(a).sum().item()))
        return h
    hs = [m.stem.register_forward_hook(make('stem'))]
    for i, st in enumerate(m.enc):
        hs.append(st.register_forward_hook(make(f'enc{i}')))
    for i, blk in enumerate(m.bn):
        hs.append(blk.register_forward_hook(make(f'bn{i}')))
    hs.append(m.bn_proj.register_forward_hook(make('bn_proj')))
    for i in range(5):
        hs.append(m.dec[i].register_forward_hook(make(f'dec{i}')))
    hs.append(m.tail.register_forward_hook(make('tail')))
    return s, hs


def seeded_inputs():
    """Same inputs every call (seed=7)."""
    g = torch.Generator().manual_seed(SEED_INPUT)
    return (torch.rand(1, 3, 64, 64, generator=g),
            torch.rand(1, 1, 64, 64, generator=g),
            torch.rand(1, 2, 64, 64, generator=g),
            torch.rand(1, 3, 64, 64, generator=g))


def fwd_with_hooks(m, s, hs):
    c, d, mv, ct = seeded_inputs()
    with torch.no_grad():
        out = m(c, d, mv, ct)
    return out, s.copy()


def build_and_fill(bias_override=None):
    """Load blob, decode, fill model. Optionally monkey-patch mx_decode_pairs to fixed bias.

    WORKAROUND for HEAD bug at semantic_fill.py:261 (`_t` undefined).
    Instead of fixing the file (read-only), we replicate fill_model in-process
    with the bn.gate zeroing step omitted (gate is initialized to zero already).
    """
    blob = open(sf.BLOB, 'rb').read()
    if bias_override is not None:
        # Monkey-patch: replace mx_decode_pairs with fixed-bias version
        # Original: w × 2^(s - median(s) - 8)
        # Override: w × 2^(s - bias_override)
        orig = sf.mx_decode_pairs

        def fixed_mx(pairs, bias=None):
            w = sf.e4m3_decode(pairs[:, 0])
            s = pairs[:, 1].astype(np.float64)
            return w * np.power(2.0, s - bias_override)
        sf.mx_decode_pairs = fixed_mx
    try:
        by_block = {}
        for name, B, off in sf.walk_records(blob):
            bn = int(name.split('.')[0][5:])
            layer = name.split('.')[1]
            raw = blob[off + 4:off + B] if B > 4 else blob[off:off + B]
            arr = np.frombuffer(raw, dtype=np.uint8)
            main, misc = sf.decode_record_raw(arr, total_B=B)
            by_block.setdefault(bn, {})[layer] = (main, misc)
        torch.manual_seed(SEED_MODEL)
        m = DLSS5NetCalib().eval()
        _fill_model_replica(m, by_block)   # local replica, avoids the HEAD bug
        return m, by_block
    finally:
        if bias_override is not None:
            sf.mx_decode_pairs = orig


def _fill_model_replica(model, by_block):
    """In-process replica of semantic_fill.fill_model, skipping the buggy _t line.

    Logic mirrors fill_model lines 87-247.  This is read-only: we do NOT touch
    semantic_fill.py.  Behaviour diverges only at the bn.gate zeroing (skipped;
    gate is already zeros by default init, so net effect is identical).
    """
    pmap = dict(model.named_parameters())
    unfilled = set(pmap)
    stats = {'filled': 0, 'pad': 0}

    def put(pname, vals, count=None):
        p = pmap[pname]
        n = p.numel()
        v = vals[:n] if count is None else vals[:count]
        if len(v) < n:
            v = np.concatenate([v, np.zeros(n - len(v), v.dtype)])
        with torch.no_grad():
            p.copy_(torch.from_numpy(v.astype(np.float32)).reshape(p.shape).to(p.dtype))
        unfilled.discard(pname)
        stats['filled'] += n
        return n

    # 单记录 Swin
    def fill_swin(prefix, main, misc):
        q = 0
        for suffix in ('attn.qkv.weight','attn.proj.weight','mlp.0.weight','mlp.2.weight'):
            pn = f'{prefix}.{suffix}'
            if pn in pmap:
                q += put(pn, main[q:])
        tail = np.concatenate([main[q:], misc]) if len(misc) else main[q:]
        o = 0
        for suffix in ('norm1.weight','norm1.bias','norm2.weight','norm2.bias',
                       'mlp.0.bias','mlp.2.bias','attn.qkv.bias','attn.proj.bias'):
            pn = f'{prefix}.{suffix}'
            if pn in pmap and pn in unfilled:
                o += put(pn, tail[o:])

    stage_blocks = {
        'enc': [[1,2,3],[5,6,7],[9,10,11,12,13],[15,16,17,18,19,20,21],
                [23,24,25,26,27,28,29]],
        'dec': [[40,41,42,43,44,45,46,47],[49,50,51,52,53,54,55],
                [57,58,59,60,61],[63,64,65],[67,68,69]],
    }
    for net in ('enc','dec'):
        for si, blocks in enumerate(stage_blocks[net]):
            for bi, b in enumerate(blocks):
                if b not in by_block: continue
                layers = by_block[b]
                if len(layers) > 1:
                    m2,_ = layers.get('layer2',(np.zeros(0),np.zeros(0)))
                    m1,s1 = layers.get('layer1',(np.zeros(0),np.zeros(0)))
                    m0,_ = layers.get('layer0',(np.zeros(0),np.zeros(0)))
                    m3,s3 = layers.get('layer3',(np.zeros(0),np.zeros(0)))
                    prefix = f'{net}.{si}.blocks.{bi}'
                    put(f'{prefix}.attn.qkv.weight', m2)
                    put(f'{prefix}.attn.proj.weight', m1)
                    mlp_stream = np.concatenate([m0, m2[786432:], m3[:262144]])
                    put(f'{prefix}.mlp.0.weight', mlp_stream)
                    put(f'{prefix}.mlp.2.weight', mlp_stream[456704:])
                    misc_all = np.concatenate([x for x in (s1, s3) if len(x)]) if (len(s1) or len(s3)) else np.zeros(0)
                    for suffix in ('norm1.weight','norm1.bias','norm2.weight','norm2.bias'):
                        pn = f'{prefix}.{suffix}'
                        if pn in pmap and pn in unfilled and len(misc_all):
                            put(pn, misc_all)
                    pn = f'{prefix}.mlp.2.bias'
                    if pn in pmap and pn in unfilled:
                        bias_src = misc_all[512:1024] if len(misc_all) >= 1024 else misc_all
                        if len(bias_src) >= 512:
                            put(pn, bias_src[:512])
                else:
                    main, misc = layers['layer0']
                    fill_swin(f'{net}.{si}.blocks.{bi}', main, misc)

    if 0 in by_block:
        main0, misc0 = by_block[0]['layer0']
        put('stem.weight', main0)
        if len(misc0): put('stem.bias', misc0)
    for mi, b in enumerate((4,8,14,22)):
        if b in by_block:
            main, misc = by_block[b]['layer0']
            put(f'merges.{mi}.reduction.weight', main)
            if len(misc):
                put(f'merges.{mi}.norm.weight', misc)
                put(f'merges.{mi}.norm.bias', misc)
    for ei, b in enumerate((48,56,62,66)):
        if b in by_block:
            main, misc = by_block[b]['layer0']
            pn = f'expands.{ei}.expand.weight'
            if pn in pmap:
                put(pn, main)
            for pn_tail in ('norm.weight','norm.bias'):
                pn = f'expands.{ei}.{pn_tail}'
                if pn in pmap and len(misc): put(pn, misc)
    for bi, b in enumerate(range(31, 39)):
        if b in by_block:
            layers = by_block[b]
            m0,_ = layers.get('layer0',(np.zeros(0),np.zeros(0)))
            m1,_ = layers.get('layer1',(np.zeros(0),np.zeros(0)))
            m2,_ = layers.get('layer2',(np.zeros(0),np.zeros(0)))
            m4 = layers.get('layer4',(np.zeros(0),np.zeros(0)))[0]  # may not exist
            # Match HEAD bn block fill
            put(f'bn.{bi}.wqkv.weight', m0)
            put(f'bn.{bi}.proj.weight', m1)
            put(f'bn.{bi}.proj.bias', m1[4194304:]) if len(m1)>4194304 else None
            put(f'bn.{bi}.side.weight', m2)
            put(f'bn.{bi}.ffwd.weight', m4) if len(m4) else None
            put(f'bn.{bi}.ffwd.bias', m4[1048576:]) if len(m4)>1048576 else None
            # bn.gate NOT zeroed here (HEAD bug); gate defaults to zero anyway
    if 39 in by_block:
        main, misc = by_block[39]['layer0']
        put('bn_proj.weight', main) if 'bn_proj.weight' in pmap else None
    if 70 in by_block:
        layers70 = by_block[70]
        main, misc = layers70.get('layer0',(np.zeros(0),np.zeros(0)))
        bs = layers70.get('blend_scale',(np.zeros(0,dtype=np.float32),np.zeros(0,dtype=np.float32)))[0]
        allv = np.concatenate([main, misc, bs]) if len(bs) else (np.concatenate([main,misc]) if len(misc) else main)
        seq = np.concatenate([allv, np.zeros(64, np.float32)])
        put('tail.global_fc.weight', seq)
        put('tail.global_fc.bias', seq[1024:])
        put('tail.conv.weight', seq[1056:])
        put('tail.conv.bias', seq[1920:])
        if 'tail.blend' in pmap:
            put('tail.blend', seq[1923:1924])

    # Fallback
    for pn in list(unfilled):
        if '_pad' in pn or pn.endswith('.pad') or pn in ('calib_pad','stem_pad'):
            unfilled.discard(pn); continue
        if 'relative_position_bias_table' in pn:
            with torch.no_grad(): pmap[pn].zero_()
            unfilled.discard(pn); continue
        if 'norm' in pn or pn.endswith(('.bias','.gate')):
            p = pmap[pn]
            with torch.no_grad():
                p.fill_(1.0 if ('norm' in pn and 'weight' in pn) else 0.0)
            unfilled.discard(pn)


def fresh_model_with_filled(by_block, seed=SEED_MODEL):
    """Build a new model with given seed and refill (used for rel_bias test)."""
    torch.manual_seed(seed)
    m = DLSS5NetCalib().eval()
    sf.fill_model(m, by_block)
    return m


# ---------- Task B (median-8 vs fixed bias) ----------
def task_b_bias_sweep():
    print('\n=== Task B: median-8 (HEAD) vs fixed bias ===')
    biases = [None, 203, 204, 205, 206, 207]   # None = use HEAD (median-8)
    rows = []
    out_table = {}
    for b in biases:
        torch.manual_seed(SEED_MODEL)
        m, by_block = build_and_fill(bias_override=b)
        s, hs = stage_hooks(m)
        out, stats = fwd_with_hooks(m, s, hs)
        label = 'median-8' if b is None else f'fixed={b}'
        am = out.abs().mean().item()
        amx = out.abs().max().item()
        rows.append((label, am, amx, stats))
        out_table[label] = (am, amx, stats)
        for h in hs: h.remove()

    # Per-stage table
    print(f'\n  {"bias":12s} | {"enc0":>10s} {"enc1":>10s} {"enc4":>10s} | {"bn0":>10s} {"bn3":>10s} {"bn7":>10s} | {"dec0":>10s} {"dec1":>10s} {"tail":>10s} | out_absmean out_absmax')
    for label, am, amx, st in rows:
        def fmt(k): return f'{st.get(k,(0,0,0))[0]:10.4e}'
        print(f'  {label:12s} | {fmt("enc0")} {fmt("enc1")} {fmt("enc4")} | {fmt("bn0")} {fmt("bn3")} {fmt("bn7")} | {fmt("dec0")} {fmt("dec1")} {fmt("tail")} | {am:.4e}  {amx:.4e}')
    return out_table


# ---------- Task A (bn sublayer) ----------
def task_a_bn_sublayers():
    print('\n=== Task A: bn0 and bn7 sublayer + rel_bias ===')
    torch.manual_seed(SEED_MODEL)
    m, _ = build_and_fill()  # HEAD (median-8)
    # Hook each sublayer (ffwd/wqkv/proj/side) of bn0 and bn7
    def hook_block(blk, label):
        st = {}
        def make(name):
            def h(mod, i, o):
                a = o.detach().float()
                st[name] = (a.abs().mean().item(), a.abs().max().item(), tuple(a.shape))
            return h
        hs = [blk.ffwd.register_forward_hook(make(f'{label}.ffwd')),
              blk.wqkv.register_forward_hook(make(f'{label}.wqkv')),
              blk.proj.register_forward_hook(make(f'{label}.proj')),
              blk.side.register_forward_hook(make(f'{label}.side'))]
        return st, hs

    # Hook ALL bn blocks to identify which stage explodes most
    full = {}
    for i, blk in enumerate(m.bn):
        st, hs = hook_block(blk, f'bn{i}')
        full[f'bn{i}'] = st
        full[f'bn{i}_hs'] = hs

    # Rel_bias impact: temporarily zero rel_bias in every SwinBlock across encoder
    # before running (we hook the model's enc+dec SwinBlocks)
    rel_stats = {}
    def make_rel_h(nm):
        def h(mod, i, o):
            a = o.detach().float()
            rel_stats[nm] = (a.abs().mean().item(), a.abs().max().item())
        return h
    rh = []
    for stage in list(m.enc) + list(m.dec):
        for blk in stage.blocks:
            rh.append(blk.attn.register_forward_hook(make_rel_h(f'{stage.dim}d')))

    s, hs = stage_hooks(m)
    out, stats = fwd_with_hooks(m, s, hs)
    for h in hs + rh: h.remove()
    for i in range(8):
        for h in full[f'bn{i}_hs']: h.remove()

    # Per-bn sublayer table
    print('  block    ffwd                wqkv                proj                side                out')
    for i in range(8):
        s_full = full[f'bn{i}']
        def fmt(k):
            v = s_full.get(f'bn{i}.{k}', (0, 0, ()))
            return f'{v[0]:9.3e}/{v[1]:9.3e}'
        out_v = stats[f'bn{i}'][0]
        print(f'  bn{i}     {fmt("ffwd")}   {fmt("wqkv")}   {fmt("proj")}   {fmt("side")}   out={out_v:.3e}')

    # rel_bias summary (note: this captures attn *output* of every SwinBlock, but
    # they all share the same hook dict — overwrites. We need a per-stage view, redo.)
    # Simpler: rerun with rel_bias=0 forcibly (set the table to 0) and compare.
    print('\n  --- rel_bias impact: fill_model zeroes rel_bias_table; test NON-ZERO effect ---')
    torch.manual_seed(SEED_MODEL)
    m2, _ = build_and_fill()
    # fill_model zeroes rel_bias_table. Set them to non-zero test value (the init).
    # Save the actual init std by checking what the model would have had: 0.02 trunc_normal_
    test_std = 0.02
    n_changed = 0
    for stage in list(m2.enc) + list(m2.dec):
        for blk in stage.blocks:
            if hasattr(blk.attn, 'relative_position_bias_table'):
                with torch.no_grad():
                    blk.attn.relative_position_bias_table.data.normal_(0, test_std)
                n_changed += 1
    print(f'  set rel_bias_table non-zero on {n_changed} SwinBlocks (std={test_std})')
    s2, hs2 = stage_hooks(m2)
    out2, stats2 = fwd_with_hooks(m2, s2, hs2)
    for h in hs2: h.remove()

    print(f'  {"stage":10s} | rel_bias=0     | rel_bias=N(0,{test_std}) | delta(%)')
    for nm in ('enc0','enc1','enc2','enc3','enc4','bn0','bn3','bn7','bn_proj','dec0','dec1','dec2','dec3','dec4','tail'):
        v1 = stats[nm][0]; v2 = stats2[nm][0]
        if v1 == 0:
            delta = 0
        else:
            delta = (v2 - v1) / v1 * 100
        print(f'  {nm:10s} | {v1:13.4e} | {v2:13.4e} | {delta:+8.2f}%')

    return stats, stats2


# ---------- Main ----------
def main():
    t0 = time.time()

    # Step 1: full stage table at HEAD (median-8, seeded)
    print('=== Task 2: seeded forward at HEAD (median-8) ===')
    print(f'  seed_model={SEED_MODEL}, seed_input={SEED_INPUT}')
    torch.manual_seed(SEED_MODEL)
    m, _ = build_and_fill()
    s, hs = stage_hooks(m)
    out, stats = fwd_with_hooks(m, s, hs)
    for h in hs: h.remove()
    print(f'  out shape={tuple(out.shape)} absmean={out.abs().mean().item():.4e} absmax={out.abs().max().item():.4e} NaN={int(torch.isnan(out).sum().item())}')
    print(f'  {"stage":10s} {"absmean":>12s} {"absmax":>12s} {"NaN":>5s}')
    for nm in ('stem','enc0','enc1','enc2','enc3','enc4','bn0','bn1','bn2','bn3','bn4','bn5','bn6','bn7','bn_proj','dec0','dec1','dec2','dec3','dec4','tail'):
        if nm in stats:
            am, amx, nn = stats[nm]
            print(f'  {nm:10s} {am:12.4e} {amx:12.4e} {nn:5d}')

    # Step 2: task A
    stats_orig, stats_zero = task_a_bn_sublayers()

    # Step 3: task B
    bias_table = task_b_bias_sweep()

    # save raw stats for docs
    out = {
        'seed_model': SEED_MODEL,
        'seed_input': SEED_INPUT,
        'baseline': {k: list(v) for k, v in stats.items()},
        'rel_bias_zeroed': {k: list(v) for k, v in stats_zero.items()},
        'bias_sweep': {label: {'out': [v[0], v[1]],
                               'stages': {k: list(vv) for k, vv in v[2].items()}}
                       for label, v in bias_table.items()},
    }
    np.save(os.path.join(HERE, '..', '.tmp', 'fwdtest_results.npy'),
            np.array(json.dumps(out), dtype=object))
    with open(os.path.join(HERE, '..', '.tmp', 'fwdtest_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[timing] total {time.time()-t0:.1f}s')
    print('[saved] .tmp/fwdtest_results.json')


if __name__ == '__main__':
    main()