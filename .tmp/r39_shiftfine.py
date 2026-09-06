import os
os.environ["DLSS5_TAIL_SIGN"]="game"; os.environ["DLSS5_MV_U_SCALE"]="-0.14"; os.environ["DLSS5_MV_V_SCALE"]="1.12"
import sys; sys.path.insert(0,"."); sys.path.insert(0,".tmp")
exec(open(".tmp/r39_shiftab.py").read().split("for name, s in")[0])
for name, s in [("s0(0,0)",(0,0)), ("s1(1,1)",(1,1)), ("s2(2,2)",(2,2)), ("s3(3,3)",(3,3)), ("s4(4,4)",(4,4)), ("s5(5,5)",(5,5)), ("s6(6,6)",(6,6))]:
    STATE["sh"] = s
    f2 = score(forward_np(m, comb[0][0]), comb[0][1], comb[0][0])
    f3 = score(forward_np(m, comb[1][0]), comb[1][1], comb[1][0])
    print(f"{name:10s}: f2={f2:+.4f} f3={f3:+.4f}", flush=True)
