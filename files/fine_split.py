"""
Participant C — fine subclustering of the myeloid compartment on A's prepared state.
Input: work/05_prepared.h5ad  (== A's prepared foton output, sha256:1891f92d...)
Markers are read from .raw (log-normalized full gene set); .X holds only scaled HVG.
Outputs: steps/20_fine_monocytes.json (result summary), steps/20_fine_clusters.csv (per-cell labels).
"""
import json
import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 1
np.random.seed(0)

ad = sc.read_h5ad("work/05_prepared.h5ad")
assert ad.raw is not None, "expected .raw with log-norm full genes"
raw_names = list(ad.raw.var_names)

def raw_mean(mask, gene):
    """mean log-norm expression of `gene` over cells in boolean mask, from .raw"""
    if gene not in raw_names:
        return None
    j = raw_names.index(gene)
    col = ad.raw.X[:, j]
    col = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else np.asarray(col).ravel()
    return float(np.mean(col[mask]))

def frac_pos(mask, gene):
    if gene not in raw_names:
        return None
    j = raw_names.index(gene)
    col = ad.raw.X[:, j]
    col = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else np.asarray(col).ravel()
    return float(np.mean(col[mask] > 0))

classical = ["CD14", "FCN1", "S100A8", "S100A9", "LYZ"]
nonclassical = ["FCGR3A", "MS4A7", "CDKN1C"]
lineage = {"T": "CD3D", "B": "MS4A1", "NK": "NKG7", "DC": "FCER1A", "Platelet": "PPBP",
           "Mono_pan": "LYZ"}

# ---------- Step A: reproduce B's coarse call to isolate the monocyte compartment ----------
sc.tl.leiden(ad, resolution=0.05, flavor="igraph", n_iterations=2, directed=False,
             key_added="coarse", random_state=0)
n_coarse = ad.obs["coarse"].nunique()
# identify the monocyte cluster: highest pan-myeloid LYZ + CD14, low CD3D
coarse_profile = {}
for cl in ad.obs["coarse"].cat.categories:
    m = (ad.obs["coarse"] == cl).values
    coarse_profile[cl] = {
        "n": int(m.sum()),
        "LYZ": raw_mean(m, "LYZ"), "CD14": raw_mean(m, "CD14"),
        "S100A8": raw_mean(m, "S100A8"), "FCGR3A": raw_mean(m, "FCGR3A"),
        "CD3D": raw_mean(m, "CD3D"), "MS4A1": raw_mean(m, "MS4A1"),
        "NKG7": raw_mean(m, "NKG7"),
    }
# monocyte cluster = max (LYZ+CD14+S100A8) with low CD3D
def mono_score(p):
    return (p["LYZ"] + p["CD14"] + p["S100A8"]) - 2 * p["CD3D"]
mono_cl = max(coarse_profile, key=lambda c: mono_score(coarse_profile[c]))
mono_mask = (ad.obs["coarse"] == mono_cl).values
print(f"[coarse] res=0.05 -> {n_coarse} clusters; monocyte cluster = {mono_cl}, n={int(mono_mask.sum())}")
print(json.dumps(coarse_profile, indent=1))

# ---------- Step B: whole-data recluster at finer resolutions (hint: res~0.2 igraph) ----------
res_sweep = [0.1, 0.15, 0.2, 0.3, 0.5]
sweep = {}
for r in res_sweep:
    key = f"fine_{r}"
    sc.tl.leiden(ad, resolution=r, flavor="igraph", n_iterations=2, directed=False,
                 key_added=key, random_state=0)
    # how many clusters overlap the coarse monocyte compartment, and their FCGR3A vs CD14
    labels_in_mono = ad.obs[key][mono_mask]
    # a cluster "belongs to" monocytes if >50% of its cells are inside mono_mask
    mono_subclusters = []
    for cl in labels_in_mono.unique():
        cl_mask = (ad.obs[key] == cl).values
        overlap = (cl_mask & mono_mask).sum() / max(cl_mask.sum(), 1)
        if overlap > 0.5 and cl_mask.sum() >= 20:
            mono_subclusters.append(cl)
    sweep[str(r)] = {"n_clusters": int(ad.obs[key].nunique()),
                     "mono_subclusters": sorted(mono_subclusters, key=int)}
    print(f"[sweep] res={r}: total={ad.obs[key].nunique()} clusters; "
          f"monocyte-compartment subclusters={sorted(mono_subclusters, key=int)}")

# pick the smallest resolution that yields >=2 monocyte subclusters
chosen_res = None
for r in res_sweep:
    if len(sweep[str(r)]["mono_subclusters"]) >= 2:
        chosen_res = r
        break
if chosen_res is None:
    chosen_res = 0.2
key = f"fine_{chosen_res}"
print(f"\n[chosen] fine resolution = {chosen_res} (key {key})")

# ---------- Step C: characterise the monocyte subclusters by markers (from .raw) ----------
mono_subs = sweep[str(chosen_res)]["mono_subclusters"]
result_clusters = {}
for cl in mono_subs:
    m = (ad.obs[key] == cl).values & mono_mask
    prof = {"n": int(m.sum())}
    for g in classical + nonclassical:
        prof[g] = round(raw_mean(m, g), 3)
        prof[g + "_fracpos"] = round(frac_pos(m, g), 3)
    result_clusters[str(cl)] = prof

# classify each monocyte subcluster classical vs non-classical
def classify(prof):
    cl_score = prof["CD14"] + prof["S100A8"] + prof["FCN1"]
    nc_score = prof["FCGR3A"] + prof["MS4A7"]
    return "classical_CD14" if cl_score > nc_score else "nonclassical_CD16_FCGR3A"
for cl in result_clusters:
    result_clusters[cl]["call"] = classify(result_clusters[cl])

calls = {result_clusters[cl]["call"] for cl in result_clusters}
two_populations = ("classical_CD14" in calls) and ("nonclassical_CD16_FCGR3A" in calls)

print("\n=== MONOCYTE SUBCLUSTER MARKER TABLE (mean log-norm from .raw) ===")
for cl, prof in result_clusters.items():
    print(cl, json.dumps(prof))
print(f"\nTWO POPULATIONS (classical + non-classical both present): {two_populations}")

summary = {
    "input_prepared_foton_output": "sha256:1891f92d1f2ca2deb2e7ee7ce9c5345e4d5679a8ac0aee0a851921118f850232",
    "n_cells_total": int(ad.n_obs),
    "coarse_res": 0.05,
    "coarse_n_clusters": int(n_coarse),
    "coarse_monocyte_cluster": str(mono_cl),
    "coarse_monocyte_n": int(mono_mask.sum()),
    "leiden_flavor": "igraph",
    "resolution_sweep": sweep,
    "chosen_fine_res": chosen_res,
    "monocyte_subclusters": result_clusters,
    "markers_classical": classical,
    "markers_nonclassical": nonclassical,
    "two_populations_resolved": bool(two_populations),
}
with open("steps/20_fine_monocytes.json", "w") as fh:
    json.dump(summary, fh, indent=2)

pd.DataFrame({"cell": ad.obs_names, "coarse": ad.obs["coarse"].values,
              f"fine_res{chosen_res}": ad.obs[key].values}).to_csv(
    "steps/20_fine_clusters.csv", index=False)
print("\nwrote steps/20_fine_monocytes.json and steps/20_fine_clusters.csv")
