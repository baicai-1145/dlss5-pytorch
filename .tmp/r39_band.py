"""R39 decisive: does the DLL band END with a NON-SHIFTED (S0) block?
Run all shift states S0..S6 but patch ONLY the LAST shifted block of each
stage to test 'tail block unshifted' hypothesis. Simpler decisive probe:
set shift=0 GLOBALLY but keep maskless — vs baseline. Already done: s0
kills f3. So not global. Test: only the FINAL block in each stage keeps
its shift; all others S0? Or the reverse: only final shifted, rest S0.
Fast approximation: two hybrids:
  H1: last block of each stage shifted (4,4), others 0
  H2: last block 0, others (4,4)
Use model.stage lists: dec.0..2, enc.0..4, expands — iterate modules."""
import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
exec(open(".tmp/r39_shiftab.py").read().split("for name, s in")[0])
def set_shifts(mode):
    # find all SwinBlock instances in order
    blocks = [mod for mod in m.modules() if isinstance(mod, SB.SwinBlock)]
    for i, blk in enumerate(blocks):
        if mode == "H1":
            on = (i % len(blocks) == len(blocks)-1)  # placeholder
    # group by stage containers
    stages = []
    for name, mod in m.named_modules():
        if isinstance(mod, SB.SwinStage):
            stages.append((name, list(mod.blocks)))
    for name, blks in stages:
        for j, blk in enumerate(blks):
            last = (j == len(blks)-1)
            if mode == "H1":
                blk.shift = last; blk.shift_size = 4 if last else 0
            elif mode == "H2":
                blk.shift = (not last); blk.shift_size = 4 if (not last) else 0
    # note: SwinBlock caches attn mask? _attn_mask computed per forward via self._attn_mask — check shift/shift_size used dynamically
for name, s0 in [("base", None), ("H1-lastShift", "H1"), ("H2-lastPlain", "H2")]:
    if s0: set_shifts(s0)
    f2 = score(forward_np(m, comb[0][0]), comb[0][1], comb[0][0])
    f3 = score(forward_np(m, comb[1][0]), comb[1][1], comb[1][0])
    print(f"{name:14s}: f2={f2:+.4f} f3={f3:+.4f}", flush=True)
