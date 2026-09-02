import sys
sys.path.insert(0, '/Users/baicai1145/dlss5-pytorch/phase3')
sys.path.insert(0, '/Users/baicai1145/dlss5-pytorch/phase4')
import numpy as np, torch
from dlss5.calib_model import DLSS5NetCalib
import semantic_fill as sf

blob = open('/Users/baicai1145/dlss5-pytorch/weights_blob.bin','rb').read()
by_block = {}
for name, B, off in sf.walk_records(blob):
    bn = int(name.split('.')[0][5:]); layer = name.split('.')[1]
    raw = blob[off+4:off+B] if B>4 else blob[off:off+B]
    arr = np.frombuffer(raw, dtype=np.uint8)
    main, misc = sf.decode_record_raw(arr, total_B=B)
    by_block.setdefault(bn, {})[layer] = (main, misc)

m = DLSS5NetCalib().eval()
pmap = dict(m.named_parameters())
sf.fill_model(m, by_block)

# Compare dec.0.blocks[0..7] and enc.4.blocks[0..6] (note enc4 has 7 blocks)
def block_stats(prefix):
    print(f'\n=== {prefix} ===')
    for i in range(8):
        blk_pref = f'{prefix}.{i}'
        if blk_pref not in [n.rsplit('.',1)[0] for n in pmap]:
            break
        for sub in ['attn.qkv.weight','attn.proj.weight','mlp.0.weight','mlp.2.weight','norm1.weight','norm1.bias','norm2.weight','norm2.bias','mlp.2.bias']:
            pn = f'{blk_pref}.{sub}'
            if pn in pmap:
                t = pmap[pn].detach()
                print(f'  {pn:42s} {tuple(t.shape)} std={t.std().item():.4f} absmean={t.abs().mean().item():.4f}')

block_stats('enc.4.blocks')
block_stats('dec.0.blocks')

# Also: hook into a single dec.0 block to see what attn + mlp produce vs the residual
print('\n--- dec.0.blocks.0 internal sublayer output ---')
blk = m.dec[0].blocks[0]
# Run a synthetic forward through just this block
device = next(m.parameters()).device
x_test = torch.randn(1, 512, 4, 4)  # bn_proj output shape (after dec0)
import torch.nn.functional as F
B,C,H,W = x_test.shape
# Manually replicate SwinBlock.forward with sub-hooks
stats = {}
for name in ['norm1','attn','norm2','mlp.0','mlp.2']:
    stats[name] = []

# Hook Linear modules inside
def make_h(name):
    def h(mod, inp, out):
        a = out.detach().float()
        stats[name].append((a.abs().mean().item(), a.abs().max().item(), tuple(a.shape)))
    return h

for name, mod in [('attn.qkv', blk.attn.qkv), ('attn.proj', blk.attn.proj), ('mlp.0', blk.mlp[0]), ('mlp.2', blk.mlp[2])]:
    mod.register_forward_hook(make_h(name))

# Hook sub-blocks
def ln_h(name):
    def h(mod, inp, out):
        a = out.detach().float()
        stats[name].append((a.abs().mean().item(), a.abs().max().item(), tuple(a.shape)))
    return h
blk.norm1.register_forward_hook(ln_h('norm1'))
blk.norm2.register_forward_hook(ln_h('norm2'))

with torch.no_grad():
    out = blk(x_test)
print(f'input  absmean={x_test.abs().mean().item():.4f}')
print(f'output absmean={out.abs().mean().item():.4f} absmax={out.abs().max().item():.4f}')
for k,v in stats.items():
    if v: print(f'  {k:10s} absmean={v[0][0]:.4f} absmax={v[0][1]:.4f} shape={v[0][2]}')