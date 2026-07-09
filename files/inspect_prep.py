import scanpy as sc
import numpy as np

ad = sc.read_h5ad("work/05_prepared.h5ad")
print("shape (cells x HVG):", ad.shape)
print("obs cols:", list(ad.obs.columns))
print("var head:", list(ad.var_names[:5]), "...")
print("has .raw:", ad.raw is not None)
if ad.raw is not None:
    print("raw shape:", ad.raw.shape)
    for g in ["CD14","CD3D","FCGR3A","LYZ","MS4A1"]:
        print("  raw has", g, ":", g in list(ad.raw.var_names))
print("obsm:", list(ad.obsm.keys()))
print("uns:", list(ad.uns.keys()))
print("neighbors in uns:", "neighbors" in ad.uns)
print("obsp:", list(ad.obsp.keys()))
# X stats
X = ad.X
print("X dtype", X.dtype, "min", float(np.min(X)), "max", float(np.max(X)))
