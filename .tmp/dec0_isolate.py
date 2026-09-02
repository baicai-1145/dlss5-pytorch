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

# Load all
m = DLSS5NetCalib().eval()
sf.fill_model(m, by_block)

# Build a forward that only runs up to a given stage
def fwd_to(stage_name):
    """stem->enc0..4->bn..->decX. Returns tensor at the requested stage."""
    color = torch.rand(1,3,64,64); depth = torch.rand(1,1,64,64)
    mvec = torch.rand(1,2,64,64); control = torch.rand(1,3,64,64)
    x = torch.cat([color, depth, mvec, control], 1)
    x = m.stem(x)
    skips = []
    for i, st in enumerate(m.enc):
        x = st(x)
        if i < len(m.enc)-1:
            skips.append(x)
            x = m.merges[i](x)
    if stage_name == 'enc4': return x
    for bi, blk in enumerate(m.bn):
        x = blk(x)
        if stage_name == f'bn{bi}': return x
    x = m.bn_proj(x)
    if stage_name == 'bn_proj': return x
    x = m.dec[0](x)
    if stage_name == 'dec0': return x
    for i in range(len(m.expands)):
        x = m.expands[i](x)
        sk = skips[-(i+1)]
        if x.shape[2:] != sk.shape[2:]:
            x = torch.nn.functional.interpolate(x, size=sk.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, sk], 1)
        lo = m.dec[i+1].dim
        x = x[:, :lo]
        x = m.dec[i+1](x)
        if stage_name == f'dec{i+1}': return x
    out = m.tail(x)
    if stage_name == 'tail': return out
    return out

with torch.no_grad():
    for s in ['enc4','bn0','bn7','bn_proj','dec0']:
        t = fwd_to(s)
        print(f'{s:8s} absmean={t.abs().mean().item():.4e} absmax={t.abs().max().item():.4e}')

# Per-block dec0 sublayer hook
print('\n--- dec.0.blocks.0..7 detailed ---')
m2 = DLSS5NetCalib().eval()
sf.fill_model(m2, by_block)
# Replicate input pipeline up to dec0 entry
color = torch.rand(1,3,64,64); depth = torch.rand(1,1,64,64)
mvec = torch.rand(1,2,64,64); control = torch.rand(1,3,64,64)
x = torch.cat([color, depth, mvec, control], 1)
x = m2.stem(x)
skips = []
for i, st in enumerate(m2.enc):
    x = st(x)
    if i < len(m2.enc)-1:
        skips.append(x)
        x = m2.merges[i](x)
for blk in m2.bn: x = blk(x)
x = m2.bn_proj(x)

# Hook each dec.0.blocks[i] for absmean of the SWINBLOCK output (which is sum of attn + mlp inside)
for i, blk in enumerate(m2.dec[0].blocks):
    stats = {}
    def make_h(idx):
        def h(mod, inp, out):
            a = out.detach().float()
            stats[idx] = (a.abs().mean().item(), a.abs().max().item())
        return h
    blk.register_forward_hook(make_h(i))
with torch.no_grad():
    _ = m2.dec[0](x)
for i in range(8):
    print(f'  dec.0.blocks.{i}: absmean={stats[i][0]:.4e} absmax={stats[i][1]:.4e}')