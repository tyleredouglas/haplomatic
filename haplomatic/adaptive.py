#!/usr/bin/env python3
"""
Adaptive windowing for error prediction.

Usage:
  haplomatic_adaptive.py \
    --snp-freqs      simul_100_freqs.csv \
    --features       features.txt \
    --model          best_model.pt \
    --sims           sims.txt \
    --regions        regions.txt \
    [--true-freq-dir true_freqs/] \
    --output         adaptive_out.csv \
    [--log-file      adaptive_windowing.log] \
    --error-threshold 0.2
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

# ─────────────────────────── Logging Setup ────────────────────────────
def setup_logging(log_file: str):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

# ──────────────────────────── I/O Helpers ─────────────────────────────

def read_list_file(path: str):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]

def read_regions_file(path: str):
    regions = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln: continue
            if ":" in ln:
                chrom,start,end = ln.split(":")
                regions.append((chrom, int(start), int(end)))
            else:
                parts = ln.split()
                if len(parts)>=3:
                    chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                    regions.append((chrom, start, end))
                else:
                    raise ValueError(f"Bad region spec: {ln}")
    return regions

# ────────────────────────── MCMC & Feature Helpers ───────────────────────

def mcmc_model(X, b):
    k     = X.shape[1]
    p     = numpyro.sample("p", dist.Dirichlet(jnp.ones(k)))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.2))
    mu    = jnp.dot(X, p)
    numpyro.sample("y_obs", dist.Normal(mu, sigma), obs=b)

def run_mcmc(X: np.ndarray, b: np.ndarray, num_warmup=40, num_samples=40):
    kernel = NUTS(mcmc_model)
    mcmc   = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1)
    mcmc.run(jax.random.PRNGKey(0), X, b, extra_fields=("diverging","num_steps"))
    samples = mcmc.get_samples(group_by_chain=True)
    extra   = mcmc.get_extra_fields(group_by_chain=True)
    p_samps = jnp.reshape(samples["p"], (-1, X.shape[1]))
    return np.array(p_samps), float(jnp.mean(extra["diverging"])), int((jnp.log2(extra["num_steps"]+1e-8).astype(int)).max())

def lsei_estimator(X, b, lb=0.0):
    k = X.shape[1]
    cons   = ({'type':'eq','fun':lambda p: p.sum()-1})
    bounds = [(lb,1.0)]*k
    p0     = np.ones(k)/k
    def obj(p): return ((X.dot(p)-b)**2).sum()
    res = minimize(obj, p0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        res = minimize(obj, p0, method='SLSQP',
                       bounds=[(0.0,1.0)]*k, constraints=cons)
    return res.x if res.success else np.full(k, np.nan)

def effective_rank(X):
    U,s,_ = np.linalg.svd(X, full_matrices=False)
    s = s/(s.sum()+1e-8)
    return float(np.exp(-np.sum(s*np.log(s+1e-8))))

# ──────────────────────────── Model Definition ──────────────────────────

class SNPTransformerEncoder(nn.Module):
    def __init__(self, n_haps, embed_dim=64, heads=4, layers=3):
        super().__init__()
        self.proj  = nn.Linear(n_haps, embed_dim)
        blk       = nn.TransformerEncoderLayer(embed_dim, heads, batch_first=True)
        self.enc  = nn.TransformerEncoder(blk, layers)
        self.pool = nn.Linear(embed_dim,1)
    def forward(self,x):
        z = self.proj(x)
        z = self.enc(z)
        w = F.softmax(self.pool(z),dim=1)
        return (w*z).sum(dim=1)

class TabularRegressor(nn.Module):
    def __init__(self, inp, hidden=(256,128), drop=0.3):
        super().__init__()
        layers, dims = [], [inp]+list(hidden)
        for i in range(len(hidden)):
            layers += [nn.Linear(dims[i],dims[i+1]), nn.ReLU(), nn.Dropout(drop)]
        layers.append(nn.Linear(dims[-1],1))
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x).squeeze(1)

class Predictor(nn.Module):
    def __init__(self, n_haps, n_tab, embed_dim=64, drop=0.3):
        super().__init__()
        self.enc = SNPTransformerEncoder(n_haps, embed_dim)
        self.reg = TabularRegressor(embed_dim+n_tab, drop=drop)
    def forward(self,Xraw,Xtab):
        z = self.enc(Xraw)
        return self.reg(torch.cat([z,Xtab],dim=1))

# ───────────────────────── Adaptive Windowing ──────────────────────────

def extract_window_features(window_df, sim, true_df):
    X = window_df[[f"B{i+1}" for i in range(X.shape[1])]].values
    b = window_df[sim].values.astype(float)
    # run MCMC once
    p_samps, divr, td = run_mcmc(X,b)
    # stats
    feats = {
        "avg_uncertainty": float(p_samps.std(axis=0).mean()),
        "avg_skew":        float(skew(p_samps,axis=0).mean()),
        "avg_kurtosis":    float(kurtosis(p_samps,axis=0).mean()),
        "snr":             float(b.mean()/b.std()) if b.std() else np.nan,
        "condition_number":float(np.linalg.cond(X)) if X.shape[0]>=X.shape[1] else np.inf,
        "min_col_distance":int(np.min([(X[:,i]!=X[:,j]).sum()
                                       for i in range(X.shape[1])
                                       for j in range(i+1,X.shape[1])])) if X.shape[1]>1 else 0,
        "avg_row_sum":     float(X.sum(axis=1).mean()),
        "row_sum_std":     float(X.sum(axis=1).std()),
        "improved_grad":   float(np.log(np.median(1/ (np.abs(np.linalg.eigvals(X.T@X)/(np.std(b)**2+1e-8))+1e-8)))),
        "effective_rank":  effective_rank(X),
        "divergence_rate": divr,
        "avg_tree_depth":  td
    }
    # LSEI + avg SNPfreq
    lsei = lsei_estimator(X,b)
    avgf = [ float(window_df.loc[window_df[f"B{i+1}"]==1,sim].mean())
             if (window_df[f"B{i+1}"]==1).any() else np.nan
             for i in range(X.shape[1]) ]
    for i in range(X.shape[1]):
        feats[f"lsei_B{i+1}"]        = float(lsei[i])
        feats[f"avg_SNPfreq_B{i+1}"] = avgf[i]
        feats[f"p_mean_B{i+1}"]      = float(p_samps.mean(axis=0)[i])
    # true-error if available
    if true_df is not None:
        mid = (window_df.pos.min()+window_df.pos.max())/2
        idx = (true_df.pos-mid).abs().idxmin()
        true_row = true_df.iloc[idx][[f"B{i+1}" for i in range(X.shape[1])]].values
        feats["True_Error"] = float(np.abs(p_samps.mean(axis=0)-true_row).sum())
    return feats

def adaptive_windowing(df, sim, true_df, model, scaler,
                       err_thr, min_w=30000, max_w=250000,
                       step_w=5000, min_snps=10, max_snps=400,
                       step_start=20000):
    logger = logging.getLogger()
    cols = scaler.feature_names_in_
    results = []
    start_pos = df.pos.min()
    end_pos   = df.pos.max()
    logger.info(f"=== sim={sim}, region={df.chrom.iloc[0]}:{start_pos}-{end_pos}")
    while start_pos < end_pos:
        # coarse scan
        best = None
        for w in [30001,50001,70001,90001,120001,150001,200001,250001]:
            e = start_pos + w
            win = df[(df.pos>=start_pos)&(df.pos<e)]
            if len(win)<min_snps: continue
            feats = extract_window_features(win, sim, true_df)
            meta  = pd.DataFrame([feats])[cols].replace([np.inf,-np.inf],np.nan)
            if meta.isna().any(axis=1).item(): continue
            Xtab  = torch.tensor(scaler.transform(meta),dtype=torch.float32).unsqueeze(0)
            M     = win[[f"B{i+1}" for i in range(model.enc.proj.in_features)]].values
            if len(M)>=max_snps: M = M[-max_snps:]
            else: M = np.vstack([np.zeros((max_snps-len(M),M.shape[1])),M])
            Xraw  = torch.tensor(M.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                pe = float(model(Xraw,Xtab).item())
            rec = dict(
                chrom=win.chrom.iloc[0],
                start=start_pos, end=e,
                Predicted_Error=pe
            )
            # copy p_mean_B*
            for i in range(M.shape[1]):
                rec[f"p_mean_B{i+1}"] = feats[f"p_mean_B{i+1}"]
            if "True_Error" in feats:
                rec["True_Error"] = feats["True_Error"]
            if best is None or pe<best["pe"]:
                best = {"pe":pe,"rec":rec}
            if pe<=err_thr:
                break
        if best is None: break
        results.append(best["rec"])
        # advance
        nxt = df[df.pos>=start_pos+step_start]
        if nxt.empty: break
        start_pos = nxt.pos.iloc[0]
    return pd.DataFrame(results)

# ───────────────────────────────────── main ───────────────────────────────────

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snp-freqs",      required=True)
    p.add_argument("--features",       required=True)
    p.add_argument("--model",          required=True)
    p.add_argument("--sims",           required=True)
    p.add_argument("--regions",        required=True)
    p.add_argument("--true-freq-dir",  default=None)
    p.add_argument("--output",         required=True)
    p.add_argument("--log-file",       default="adaptive_windowing.log")
    p.add_argument("--error-threshold",type=float,required=True)
    args = p.parse_args()

    logger = setup_logging(args.log_file)
    # 1) load model+scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.model, map_location=device, weights_only=False)
    model  = Predictor(n_haps=8, n_tab=len(ckpt["scaler_mean"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    scaler = StandardScaler(); scaler.mean_=ckpt["scaler_mean"]
    scaler.scale_=ckpt["scaler_scale"]
    scaler.feature_names_in_=np.array(read_list_file(args.features))

    # 2) load SNP‐freqs; unify chrom col
    snp_df = pd.read_csv(args.snp_freqs)
    if "CHROM" in snp_df.columns:
        snp_df = snp_df.rename(columns={"CHROM":"chrom"})
    elif "chrom" not in snp_df.columns:
        raise KeyError("Need 'chrom' or 'CHROM' column in SNP‐freqs")

    sims    = read_list_file(args.sims)
    regions = read_regions_file(args.regions)

    all_out = []
    for chrom,start,end in regions:
        sub = snp_df.query("chrom==@chrom & @start<=pos<=@end").reset_index(drop=True)
        if sub.empty:
            logger.warning(f"No SNPs in region {chrom}:{start}-{end}")
            continue
        for sim in sims:
            true_df = None
            if args.true_freq_dir:
                tf = Path(args.true_freq_dir)/f"{sim}_true_freqs.csv"
                if tf.exists():
                    true_df = pd.read_csv(tf).rename(columns={"pos":"pos"}) 
            df_out = adaptive_windowing(
                sub, sim, true_df, model, scaler,
                err_thr=args.error_threshold
            )
            df_out["sim"] = sim
            df_out["region"] = f"{chrom}:{start}-{end}"
            all_out.append(df_out)

    if all_out:
        pd.concat(all_out, ignore_index=True).to_csv(args.output, index=False)
        logger.info(f"Wrote {len(pd.concat(all_out))} windows to {args.output}")
    else:
        logger.warning("No output generated.")

if __name__=="__main__":
    main()
