"""
Task 2c — data-drift analysis: synthetic (training) vs Chlamydomonas in-situ (test).

Quantifies how the two datasets differ, to explain why the model scores F1 0.76 on the
synthetic data but detects nothing on the in-situ chunk. Each comparison is a figure:

  1. Intensity histograms (raw + normalized)   -> the distributions differ, but normalization
                                                  equalizes the scale (rules scale out)
  2. Distribution shape (skew / kurtosis)       -> synthetic is sharp-spike; in-situ is uniform
  3. Radial power spectrum                       -> different frequency content
  4. Local contrast at ribosomes vs background   -> ribosomes stand out 2.7x vs 1.2x
  5. Nearest-neighbour spacing                   -> crowding
  6. Model ribosome-confidence distribution      -> the failure signal (0.998 vs ~0.008)
  7. 2D FFT power spectra + missing-wedge metric -> ~10x more anisotropic in-situ

Self-contained: reads submission/data (run task2a/download.py first) and the cached
ribosome heatmap if task2c produced one. Writes figures/ + drift_stats.json beside this file.
"""

import json
import numpy as np
import zarr
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # submission/
DATA = ROOT / "data"
SYN_GT = DATA / "synthetic/overlay/ribosome_annotations_synthetic.json"
CHL_GT = DATA / "chlamydomonas/overlay/ribosome_annotations_chlamydomonas.json"
# Optional cached ribosome heatmap from task2c (if you generated one); safe if missing.
HEAT_NPY = Path(__file__).resolve().parent / "heatmap_ribosome.npy"
# Predictions CSV from task2b (synthetic) — used for the confidence plot if present.
SYN_PRED_CSV = ROOT / "task2b/results/synthetic/val_pred_df_seed.csv"
# Chunk zarr produced by task2c/inference_chlamy.py (else extracted on the fly).
CHUNK_ZARR = Path(__file__).resolve().parent / "chunk_data/denoised.zarr"

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(parents=True, exist_ok=True)
VS_SYN, VS_CHL = 10.012, 7.84
X_OFF, Y_OFF, CHUNK = 160, 752, 256

stats = {}


def _glob_zarr(subdir):
    zs = list((DATA / subdir).rglob("*.zarr"))
    if not zs:
        raise FileNotFoundError(f"No .zarr under {DATA/subdir} — run task2a/download.py first.")
    return zs[0]


def load_synthetic():
    a = zarr.open(str(_glob_zarr("synthetic/tomograms")), mode="r")
    a = a["0"] if "0" in a else a
    return np.asarray(a, dtype="float32")


def load_chunk():
    """Use the task2c chunk if present; otherwise extract it from the full tomogram."""
    if CHUNK_ZARR.exists():
        a = zarr.open(str(CHUNK_ZARR), mode="r")
        a = a["0"] if "0" in a else a
        return np.asarray(a, dtype="float32")
    full = zarr.open(str(_glob_zarr("chlamydomonas/tomograms")), mode="r")
    full = full["0"] if "0" in full else full
    return np.asarray(full[:, Y_OFF:Y_OFF+CHUNK, X_OFF:X_OFF+CHUNK], dtype="float32")


def load_gt(path, vs, chunk=False):
    d = json.load(open(path))
    pts = np.array([[p["z"]/vs, p["y"]/vs, p["x"]/vs] for p in d["points"]])
    if chunk:
        m = ((pts[:,2]>=X_OFF)&(pts[:,2]<X_OFF+CHUNK)&(pts[:,1]>=Y_OFF)&(pts[:,1]<Y_OFF+CHUNK))
        pts = pts[m]; pts[:,2]-=X_OFF; pts[:,1]-=Y_OFF
    return pts


def zscore(v): return (v - v.mean()) / v.std()


print("Loading datasets...")
syn = load_synthetic()
chl = load_chunk()
syn_gt = load_gt(SYN_GT, VS_SYN)
chl_gt = load_gt(CHL_GT, VS_CHL, chunk=True)
print(f"  synthetic {syn.shape}, in-situ chunk {chl.shape}")


# ----------------------------------------------------------------------------- 1. histograms
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].hist(syn.ravel(), bins=200, density=True, color="C0"); ax[0].set_yscale("log")
ax[0].set_title("Synthetic — raw intensity"); ax[0].set_xlabel("voxel value (a.u.)")
ax[0].set_ylabel("density (log)"); ax[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax[1].hist(chl.ravel(), bins=200, density=True, color="C3"); ax[1].set_yscale("log")
ax[1].set_title("In-situ — raw intensity"); ax[1].set_xlabel("voxel value (a.u.)"); ax[1].set_ylabel("density (log)")
ax[2].hist(zscore(syn).ravel(), bins=200, range=(-6,6), density=True, alpha=.6, label="synthetic", color="C0")
ax[2].hist(zscore(chl).ravel(), bins=200, range=(-6,6), density=True, alpha=.6, label="in-situ", color="C3")
ax[2].set_title("Normalized (what the model sees)"); ax[2].set_xlabel("z-score"); ax[2].set_ylabel("density"); ax[2].legend()
fig.suptitle("Intensity: raw scales differ, but normalization equalizes the range", fontsize=11)
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(FIG/"01_histograms.png", dpi=130); plt.close()

zs, zc = zscore(syn), zscore(chl)
stats["zscored_p1_p99"] = {"synthetic": [float(np.percentile(zs,1)), float(np.percentile(zs,99))],
                           "in_situ":   [float(np.percentile(zc,1)), float(np.percentile(zc,99))]}


# ----------------------------------------------------------------------------- 2. distribution shape
from scipy.stats import skew, kurtosis
syn_m = [float(skew(syn.ravel())), float(kurtosis(syn.ravel()))]
chl_m = [float(skew(chl.ravel())), float(kurtosis(chl.ravel()))]
fig, ax = plt.subplots(figsize=(7,4.5))
x = np.arange(2); w = .35
ax.bar(x-w/2, syn_m, w, label="synthetic", color="C0")
ax.bar(x+w/2, chl_m, w, label="in-situ", color="C3")
ax.set_xticks(x); ax.set_xticklabels(["skew", "kurtosis"])
ax.set_title("Distribution shape (sharp spikes vs uniform texture)"); ax.legend()
plt.tight_layout(); plt.savefig(FIG/"02_shape.png", dpi=130); plt.close()
stats["distribution_shape"] = {"synthetic": {"skew": syn_m[0], "kurtosis": syn_m[1]},
                               "in_situ":   {"skew": chl_m[0], "kurtosis": chl_m[1]}}


# ----------------------------------------------------------------------------- 3. radial power spectrum
def radial_power(slice2d):
    f = np.fft.fftshift(np.abs(np.fft.fft2(slice2d))**2)
    cy, cx = np.array(f.shape)//2
    y, x = np.indices(f.shape)
    r = np.sqrt((x-cx)**2 + (y-cy)**2).astype(int)
    return np.bincount(r.ravel(), f.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
ps_syn = radial_power(zscore(syn)[syn.shape[0]//2])
ps_chl = radial_power(zscore(chl)[chl.shape[0]//2])
fig, ax = plt.subplots(figsize=(7,4.5))
ax.semilogy(np.linspace(0,1,len(ps_syn)), ps_syn, label="synthetic", color="C0")
ax.semilogy(np.linspace(0,1,len(ps_chl)), ps_chl, label="in-situ", color="C3")
ax.set_title("Radial power spectrum (mid slice)"); ax.set_xlabel("normalized frequency")
ax.set_ylabel("power (log)"); ax.legend()
plt.tight_layout(); plt.savefig(FIG/"03_power_spectrum.png", dpi=130); plt.close()


# ----------------------------------------------------------------------------- 4. local contrast at GT vs background
def local_contrast(vol, pts, r=8, nz=2):
    vn = zscore(vol); out=[]
    for z,y,x in pts.astype(int):
        if 0<=z<vol.shape[0] and 0<=y<vol.shape[1] and 0<=x<vol.shape[2]:
            sub = vn[max(0,z-nz):z+nz+1, max(0,y-r):y+r, max(0,x-r):x+r]
            if sub.size: out.append(max(abs(sub.min()), abs(sub.max())))
    return np.array(out)
rng = np.random.default_rng(0)
def rand_pts(vol, n):
    return np.column_stack([rng.uniform(0,vol.shape[0],n), rng.uniform(0,vol.shape[1],n), rng.uniform(0,vol.shape[2],n)])
syn_gtc = local_contrast(syn, syn_gt); syn_bgc = local_contrast(syn, rand_pts(syn,300))
chl_gtc = local_contrast(chl, chl_gt); chl_bgc = local_contrast(chl, rand_pts(chl,300))
fig, ax = plt.subplots(1, 2, figsize=(13,4.5))
for a,(gtc,bgc,name) in zip(ax, [(syn_gtc,syn_bgc,"synthetic"),(chl_gtc,chl_bgc,"in-situ")]):
    a.hist(bgc, bins=30, alpha=.6, label="background", color="gray", density=True)
    a.hist(gtc, bins=30, alpha=.6, label="ribosome", color="C2", density=True)
    ratio = np.median(gtc)/max(np.median(bgc),1e-6)
    a.set_title(f"{name}: ribosome stands out {ratio:.2f}x"); a.set_xlabel("local contrast"); a.legend()
plt.tight_layout(); plt.savefig(FIG/"04_contrast.png", dpi=130); plt.close()
stats["ribosome_vs_background_contrast_ratio"] = {
    "synthetic": float(np.median(syn_gtc)/max(np.median(syn_bgc),1e-6)),
    "in_situ":   float(np.median(chl_gtc)/max(np.median(chl_bgc),1e-6))}


# ----------------------------------------------------------------------------- 5. crowding
from scipy.spatial.distance import cdist
def nn_dist(pts, vs):
    if len(pts) < 2: return np.array([])
    D = cdist(pts, pts); np.fill_diagonal(D, np.inf)
    return D.min(1) * vs
syn_full = load_gt(SYN_GT, VS_SYN); chl_full = load_gt(CHL_GT, VS_CHL)
syn_nn = nn_dist(syn_full, VS_SYN); chl_nn = nn_dist(chl_full, VS_CHL)
fig, ax = plt.subplots(figsize=(7,4.5))
ax.hist(syn_nn, bins=30, alpha=.6, label=f"synthetic (n={len(syn_full)})", color="C0", density=True)
ax.hist(chl_nn, bins=30, alpha=.6, label=f"in-situ (n={len(chl_full)})", color="C3", density=True)
ax.set_title("Ribosome nearest-neighbour distance (crowding)"); ax.set_xlabel("distance (Å)"); ax.legend()
plt.tight_layout(); plt.savefig(FIG/"05_crowding.png", dpi=130); plt.close()
stats["n_ribosomes"] = {"synthetic": len(syn_full), "in_situ": len(chl_full)}


# ----------------------------------------------------------------------------- 6. model confidence
fig, ax = plt.subplots(figsize=(7,4.5))
import pandas as pd
if SYN_PRED_CSV.exists():
    sdf = pd.read_csv(SYN_PRED_CSV); sdf = sdf[sdf.particle_type=="ribosome"]
    if len(sdf):
        ax.hist(sdf.conf, bins=30, alpha=.7, label=f"synthetic (max={sdf.conf.max():.3f})", color="C0")
        stats["synthetic_conf_max"] = float(sdf.conf.max())
if HEAT_NPY.exists():
    heat = np.load(HEAT_NPY)
    ax.hist(np.sort(heat.ravel())[::-1][:5000], bins=30, alpha=.7,
            label=f"in-situ (max={heat.max():.3f})", color="C3")
    stats["in_situ_conf_max"] = float(heat.max())
ax.axvline(0.19, color="k", ls="--", label="detection threshold 0.19")
ax.set_title("Model ribosome confidence (the failure signal)")
ax.set_xlabel("confidence"); ax.set_yscale("log"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG/"06_confidence.png", dpi=130); plt.close()


# ----------------------------------------------------------------------------- 7. 2D FFT + missing-wedge metric
def fft2_logmag(s): return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(s))))
def central_xz(vol): return vol[:, vol.shape[1]//2, :]
def nyq(vs): return 0.5/vs
def add_spec(ax, img, vr, vc, title, rl, cl):
    ext = [-nyq(vc), nyq(vc), -nyq(vr), nyq(vr)]
    im = ax.imshow(img, cmap="magma", extent=ext, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=10); ax.set_xlabel(f"{cl} (1/Å)", fontsize=9); ax.set_ylabel(f"{rl} (1/Å)", fontsize=9)
    ax.tick_params(labelsize=7); return im
fig, ax = plt.subplots(2, 2, figsize=(12, 11))
add_spec(ax[0,0], fft2_logmag(zscore(syn)[syn.shape[0]//2]), VS_SYN, VS_SYN, f"Synthetic — XY (voxel {VS_SYN} Å)", "k_y","k_x")
add_spec(ax[0,1], fft2_logmag(zscore(chl)[chl.shape[0]//2]), VS_CHL, VS_CHL, f"In-situ — XY (voxel {VS_CHL} Å)", "k_y","k_x")
add_spec(ax[1,0], fft2_logmag(central_xz(zscore(syn))), VS_SYN, VS_SYN, "Synthetic — XZ (near-isotropic)", "k_z","k_x")
im = add_spec(ax[1,1], fft2_logmag(central_xz(zscore(chl))), VS_CHL, VS_CHL, "In-situ — XZ (missing wedge)", "k_z","k_x")
cb = fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.025, pad=0.02); cb.set_label("log(1+|F(k)|)", fontsize=9)
fig.suptitle("2D Fourier power spectra (DC at center; axes in 1/Å)", fontsize=12, y=0.98)
plt.savefig(FIG/"07_fft.png", dpi=130, bbox_inches="tight"); plt.close()

def wedge_anis(vol):
    sp = np.abs(np.fft.fftshift(np.fft.fft2(central_xz(zscore(vol)))))**2
    cz, cx = np.array(sp.shape)//2
    return float(sp[cz-2:cz+3, :].mean() / max(sp[:, cx-2:cx+3].mean(), 1e-9))
stats["xz_anisotropy_ratio"] = {"synthetic": wedge_anis(syn), "in_situ": wedge_anis(chl),
                                "note": "higher = stronger missing-wedge asymmetry"}


json.dump(stats, open(Path(__file__).resolve().parent / "drift_stats.json", "w"), indent=2)
print("\n=== DRIFT SUMMARY ===")
print(json.dumps(stats, indent=2))
print(f"\nFigures -> {FIG}/   Stats -> drift_stats.json")
