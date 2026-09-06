import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
import numpy as np, torch, torch.nn.functional as F
import dlss5, eval_suite as ES
dev="cuda:0"
import dlss5.swin_block as SB
orig_attn = None
# find the attention score->softmax site
import inspect
src = inspect.getsource(SB)
import re
for m in re.finditer(r"def (\w+)\(", src):
    pass
# monkeypatch: wrap F.softmax call inside attention via module attribute
class ClampCtx:
    def __init__(self, limit):
        self.limit = limit
    def __enter__(self):
        self.orig = torch.nn.functional.softmax
        torch.nn.functional.softmax = self.orig  # placeholder
        return self
    def __exit__(self, *a):
        pass
# simpler: patch the block class forward — locate it
print([n for n,o in inspect.getmembers(SB, inspect.isclass)])
