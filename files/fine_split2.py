"""
Participant C — focused subclustering of the monocyte compartment.
Isolate coarse cluster 3 (B's single 'Monocytes' population, n~1060), rebuild the kNN graph on
A's PCA coordinates restricted to those cells, Leiden at finer resolution, and test the
classical (CD14/FCN1/S100A8/S100A9/LYZ) vs non-classical (FCGR3A/MS4A7/CDKN1C) marker split.
All marker means are read from .raw (log-normalized full genes).
"""
import json
import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 1
np.random.seed(0)

ad = sc.read_h5ad("work/05_prepared.h5ad")
raw_names = list(ad.raw.var_names)

def rmean(mask, gene):
    j = raw_names.index(gene)
    col = ad.raw.X[:, j]
    col = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else np.asarray(col).ravel()
    return float(np.mean(col[mask]))

def rfrac(mask, gene):
    j = raw_names.index(gene)
    col = ad.raw.X[:, j]
    col = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else np.asarray(col).ravel()
    return float(np.mean(col[mask] > 0))

classical = ["CD14", "FCN1", "S100A8", "S100A9", "LYZ"]
nonclassical = ["FCGR3A", "MS4A7", "CDKN1C"]

# reproduce B's coarse call, isolate the monocyte cluster
sc.tl.leiden(ad, resolution=0.05, flavor="igraph", n_iterations=2, directed=False,
             key_added="coarse", random_state=0)
def mono_score(cl):
    m = (ad.obs["coarse"] == cl).values
    return rmean(m, "LYZ") + rmean(m, "CD14") + rmean(m, "S100A8") - 2 * rmean(m, "CD3D")
mono_cl = max(ad.obs["coarse"].cat.categories, key=mono_score)
mono_mask = (ad.obs["coarse"] == mono_cl).values
n_mono = int(mono_mask.sum())
print(f"coarse monocyte cluster {mono_cl}: n={n_mono}")

# focused subcluster: restrict to monocytes, rebuild kNN on A's PCA coords, Leiden
mono = ad[mono_mask].copy()
sc.pp.neighbors(mono, n_neighbors=15, n_pcs=30, use_rep="X_pca", random_state=0)

def characterize(labels, cells_mask_global):
    """labels: per-mono-cell cluster labels (Series aligned to mono.obs_names)."""
    out = {}
    global_idx = np.where(cells_mask_global)[0]
    for cl in sorted(labels.unique(), key=lambda x: int(x)):
        sub_local = (labels == cl).values
        gmask = np.zeros(ad.n_obs, dtype=bool)
        gmask[global_idx[sub_local]] = True
        prof = {"n": int(sub_local.sum())}
        for g in classical + nonclassical:
            prof[g] = round(rmean(gmask, g), 3)
        for g in ["CD14", "S100A8", "FCGR3A", "MS4A7"]:
            prof[g + "_fp"] = round(rfrac(gmask, g), 3)
        cl_s = prof["CD14"] + prof["S100A8"] + prof["FCN1"]
        nc_s = prof["FCGR3A"] + prof["MS4A7"] + prof["CDKN1C"]
        prof["call"] = "classical_CD14" if cl_s > nc_s else "nonclassical_CD16_FCGR3A"
        out[cl] = prof
    return out

results = {}
for r in [0.2, 0.3, 0.4]:
    key = f"sub_{r}"
    sc.tl.leiden(mono, resolution=r, flavor="igraph", n_iterations=2, directed=False,
                 key_added=key, random_state=0)
    prof = characterize(mono.obs[key], mono_mask)
    calls = {prof[c]["call"] for c in prof}
    two = ("classical_CD14" in calls) and ("nonclassical_CD16_FCGR3A" in calls)
    results[str(r)] = {"n_subclusters": len(prof), "two_populations": bool(two), "clusters": prof}
    print(f"\n--- subcluster res={r}: {len(prof)} clusters, two_populations={two} ---")
    for c, p in prof.items():
        print(c, json.dumps(p))

# choose the smallest res that cleanly yields exactly the two monocyte states
chosen = None
for r in [0.2, 0.3, 0.4]:
    if results[str(r)]["two_populations"]:
        chosen = r
        break
print(f"\nCHOSEN focused resolution: {chosen}")

summary = {
    "input_prepared_foton_output": "sha256:1891f92d1f2ca2deb2e7ee7ce9c5345e4d5679a8ac0aee0a851921118f850232",
    "prepared_foton_id": "sha256:f90b314cf5588dd03ddb22e02cf1b99ffb54185b0ba0586de7d31b0fe34384b1",
    "approach": "focused subcluster of B's coarse monocyte cluster; kNN rebuilt on A's X_pca (30 PCs), Leiden igraph",
    "coarse_monocyte_cluster": str(mono_cl),
    "coarse_monocyte_n": n_mono,
    "markers_classical": classical,
    "markers_nonclassical": nonclassical,
    "resolutions": results,
    "chosen_res": chosen,
    "two_populations_resolved": bool(chosen is not None),
}
with open("steps/21_fine_monocytes.json", "w") as fh:
    json.dump(summary, fh, indent=2)

if chosen is not None:
    key = f"sub_{chosen}"
    pd.DataFrame({"cell": mono.obs_names, "subcluster": mono.obs[key].values}).to_csv(
        "steps/21_monocyte_subclusters.csv", index=False)
print("wrote steps/21_fine_monocytes.json and steps/21_monocyte_subclusters.csv")
