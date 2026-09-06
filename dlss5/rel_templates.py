import numpy as np
S = 15
def _build():
    i, j = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    t = {"manh": (np.abs(i-7)+np.abs(j-7)).ravel(),
         "row":  np.abs(i-7).ravel(),
         "col":  np.abs(j-7).ravel(),
         "cheb": np.maximum(np.abs(i-7), np.abs(j-7)).ravel(),
         "sq":   ((i-7)**2+(j-7)**2).ravel()}
    return {k: ((v-v.mean())/(v.std()+1e-9)).astype(np.float32) for k,v in t.items()}
TEMPLATES = _build()
