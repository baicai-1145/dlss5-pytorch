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

def block_stats(prefix, n_blocks):
    print(f'\n=== {prefix} ({n_blocks} blocks) ===')
    for i in range(n_blocks):
        print(f'  block {i}:')
        for sub in ['attn.qkv.weight','attn.proj.weight','mlp.0.weight','mlp.2.weight','norm1.weight','norm1.bias','norm2.weight','norm2.bias','mlp.2.bias']:
            pn = f'{prefix}.{i}.{sub}'
            if pn in pmap:
                t = pmap[pn].detach()
                print(f'    {pn:50s} {tuple(t.shape)} std={t.std().item():.4f} absmean={t.abs().mean().item():.4f}')

block_stats('enc.4.blocks', 7)
block_stats('dec.0.blocks', 8)

# Synthetic single-block forward with sublayer hooks
print('\n--- dec.0.blocks.0 internal sublayer output (synthetic x_test) ---')
blk = m.dec[0].blocks[0]
stats = {}
def make_h(name):
    def h(mod, inp, out):
        a = out.detach().float()
        stats[name] = (a.abs().mean().item(), a.abs().max().item(), tuple(a.shape))
    return h
for name, mod in [('attn.qkv', blk.attn.qkv), ('attn.proj', blk.attn.proj),
                  ('mlp.0', blk.mlp[0]), ('mlp.2', blk.mlp[2])]:
    mod.register_forward_hook(make_h(name))
blk.norm1.register_forward_hook(make_h('norm1'))
blk.norm2.register_forward_hook(make_h('norm2'))

x_test = torch.randn(1, 512, 4, 4)
with torch.no_grad():
    out = blk(x_test)
print(f'input  absmean={x_test.abs().mean().item():.4f}')
print(f'output absmean={out.abs().mean().item():.4f} absmax={out.abs().max().item():.4f}')
for k in ('norm1','attn.qkv','attn.proj','norm2','mlp.0','mlp.2'):
    if k in stats:
        v = stats[k]
        print(f'  {k:10s} absmean={v[0]:.4f} absmax={v[1]:.4f} shape={v[2]}')