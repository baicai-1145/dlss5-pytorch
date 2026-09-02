import sys
sys.path.insert(0, '/Users/baicai1145/dlss5-pytorch/phase3')
sys.path.insert(0, '/Users/baicai1145/dlss5-pytorch/phase4')
from semantic_fill import walk_records, MX_BOUND
blob = open('/Users/baicai1145/dlss5-pytorch/weights_blob.bin','rb').read()
count = 0
for name, B, off in walk_records(blob):
    if name.startswith('b23.') or name.startswith('b40.') or name.startswith('b47.'):
        print(f'{name} B={B} raw_len={B-4 if B>4 else B} in_MX_BOUND={B in MX_BOUND}')
        count += 1
        if count >= 6: break