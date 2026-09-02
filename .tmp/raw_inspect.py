import sys
sys.path.insert(0, '/Users/baicai1145/dlss5-pytorch/phase3')
sys.path.insert(0, '/Users/baicai1145/dlss5-pytorch/phase4')
import numpy as np
from semantic_fill import walk_records, decode_record_raw, e4m3_decode, mx_decode_pairs, MX_BOUND

blob = open('/Users/baicai1145/dlss5-pytorch/weights_blob.bin','rb').read()
for name, B, off in walk_records(blob):
    if name in ('block40.layer0.layer','block40.layer2.layer','block40.layer3.layer',
                'block41.layer0.layer','block41.layer2.layer','block41.layer3.layer',
                'block23.layer0.layer','block23.layer2.layer','block23.layer3.layer'):
        raw = blob[off+4:off+B] if B>4 else blob[off:off+B]
        arr = np.frombuffer(raw, dtype=np.uint8)
        main, misc = decode_record_raw(arr, total_B=B)
        # Print decoded main statistics by segment
        if B == 917568:
            lo, hi = MX_BOUND[B]
            seg_e4a = e4m3_decode(arr[:lo])
            seg_mx  = mx_decode_pairs(arr[lo:hi].reshape(-1,2))
            seg_e4b = e4m3_decode(arr[hi:])
            print(f'{name} B={B} e4a[{len(seg_e4a)}]={seg_e4a.std():.4f}/abs{abs(seg_e4a).mean():.4f} mx[{len(seg_mx)}]={seg_mx.std():.4f}/abs{abs(seg_mx).mean():.4f} e4b[{len(seg_e4b)}]={seg_e4b.std():.4f}/abs{abs(seg_e4b).mean():.4f}')
        else:
            print(f'{name} B={B} main[{len(main)}]={main.std():.4f}/abs{abs(main).mean():.4f} misc[{len(misc)}]={misc.std():.4f}/abs{abs(misc).mean():.4f}')