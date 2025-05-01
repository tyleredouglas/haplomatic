#!/usr/bin/env python3
"""
Slide fixed-size windows across SNP-frequency tables and compute a
feature/error table for one or many populations, checkpointing every
2 000 windows to guard against preemption—and resuming from any
already-written windows in the output CSV on re-launch.

• Populations:  one sim name per line in --populations  
• Founders:     one founder name per line in --founders-file  
• SNP freqs:    CSV with columns CHROM,pos,<founders…>,<pop1>,<pop2>…  
• True freqs:   files <pop>_true_freqs.csv in --true-freq-dir  
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew

# ─────────────────────────── utility small bits ──────────────────────────── #

def read_list_file(path: str) -> List[str]:
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]

def lsei_haplotype_estimator(X: np.ndarray, b: np.ndarray,
                             lower_bound: float = 0.0) -> np.ndarray:
    k = X.shape[1]
    cons   = ({'type':'eq','fun':lambda p: p.sum()-1})
    bounds = [(lower_bound,1.0)]*k
    p0     = np.ones(k)/k
    def obj(p): return ((X@p - b)**2).sum()

    res = minimize(obj, p0, method="SLSQP",
                   bounds=bounds, constraints=cons)
    if not res.success:
        res = minimize(obj, p0, method="SLSQP",
                       bounds=[(0.0,1.0)]*k,
                       constraints=cons)
    return res.x if res.success else np.full(k, np.nan)

def compute_effective_rank(X: np.ndarray) -> float:
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    s_norm = s/(s.sum()+1e-8)
    ent    = -np.sum(s_norm*np.log(s_norm+1e-8))
    return float(np.exp(ent))


# ─────────────────────────── window generation ───────────────────────────── #

class FixedWindowGenerator:
    """
    Slide every size in window_sizes_kb (converted to bp) across the SNP table
    with stride_kb, collecting SNP rows for each window.
    """
    def __init__(self,
                 window_sizes_kb: Tuple[int, ...],
                 stride_kb: int = 20,
                 min_snps_per_window: int = 10):
        self.window_sizes_bp = [sz*1000 for sz in window_sizes_kb]
        self.stride_bp       = stride_kb*1000
        self.min_snps        = min_snps_per_window

    def generate(self,
                 observed: pd.DataFrame,
                 true_freqs: pd.DataFrame,
                 founder_cols: List[str],
                 sim_col: str
                 ) -> Dict[Tuple[str, Tuple[int,int]], Dict]:
        results: Dict = {}
        obs = observed.sort_values("pos").reset_index(drop=True)
        min_pos, max_pos = obs["pos"].iloc[[0,-1]]
        anchor = min_pos
        while anchor + self.window_sizes_bp[0] <= max_pos:
            for size in self.window_sizes_bp:
                start, end = anchor, anchor + size
                if end > max_pos: continue
                win_df = obs.query("@start <= pos < @end").reset_index(drop=True)
                if len(win_df) < self.min_snps: continue
                mid_idx = (true_freqs.pos - win_df.pos.median()).abs().idxmin()
                true_row = true_freqs.loc[mid_idx, founder_cols].values
                results[(sim_col,(start,end))] = {
                    "window": win_df,
                    "true_freq_row": true_row,
                    "window_size_bp": size
                }
            anchor += self.stride_bp
        return results


# ───────────────────────────── Feature builder ───────────────────────────── #

class FeatureBuilder:
    """
    Builds the feature/error records using a single NUTS chain.
    """
    def __init__(self,
                 num_warmup: int = 100,
                 num_samples: int = 100,
                 rng_seed: int = 0):
        self.num_warmup  = num_warmup
        self.num_samples = num_samples
        self.rng_seed    = rng_seed

    def _posterior_p(self, X: np.ndarray, b: np.ndarray):
        k = X.shape[1]
        def _model(X_,b_obs):
            p     = numpyro.sample("p", dist.Dirichlet(jnp.ones(k)))
            sigma = numpyro.sample("sigma", dist.HalfNormal(0.2))
            numpyro.sample("y", dist.Normal(jnp.dot(X_,p), sigma), obs=b_obs)

        mcmc = MCMC(NUTS(_model),
                    num_warmup=self.num_warmup,
                    num_samples=self.num_samples,
                    num_chains=1)
        mcmc.run(jax.random.PRNGKey(self.rng_seed), X, b,
                 extra_fields=("diverging","num_steps"))
        samp  = mcmc.get_samples(group_by_chain=True)
        extra = mcmc.get_extra_fields(group_by_chain=True)

        p_draw   = np.asarray(samp["p"]).reshape(-1,k)
        sigma    = np.asarray(samp["sigma"]).reshape(-1)
        div_rate = float(np.mean(extra["diverging"]))
        max_td   = int(np.log2(extra["num_steps"]+1e-8).astype(int).max())
        return p_draw, sigma, div_rate, max_td

    def build(self,
              windows: Dict[Tuple[str, Tuple[int,int]], Dict],
              founder_cols: List[str],
              sim_col: str) -> pd.DataFrame:
        recs = []
        for (pop,(start,end)), info in windows.items():
            win = info["window"]
            X   = win[founder_cols].to_numpy(float)
            b   = win[sim_col].to_numpy(float)

            p_draw, sigma_draw, div_rt, td = self._posterior_p(X,b)
            p_mean = p_draw.mean(axis=0)
            p_std  = p_draw.std(axis=0)

            # core features
            avg_unc = p_std.mean()
            avg_skw = skew(p_draw).mean()
            avg_krt = kurtosis(p_draw).mean()
            sig     = sigma_draw.mean() if sigma_draw.size else 1.0
            H       = (X.T@X)/(sig**2+1e-8)
            imp_g   = float(np.log(np.median(1/(np.abs(np.linalg.eigvals(H))+1e-8))+1e-8))
            err     = float(np.abs(p_mean - info["true_freq_row"]).sum())

            # LSEI + avg SNP‐freq
            lsei = lsei_haplotype_estimator(X,b)
            avgf = np.array([
                win.loc[win[c]==1,sim_col].mean()
                if (win[c]==1).any() else np.nan
                for c in founder_cols
            ])

            rec = dict(
              sim=pop,
              chrom=win["CHROM"].iloc[0],
              start=start,
              end=end,
              window_bp=end-start+1,
              n_snps=len(win),
              avg_uncertainty=avg_unc,
              avg_skew=avg_skw,
              avg_kurtosis=avg_krt,
              snr=float(b.mean()/b.std()) if b.std() else np.nan,
              condition_number=float(np.linalg.cond(X)) if X.shape[0]>=X.shape[1] else np.inf,
              min_col_distance=int(np.min([
                  (X[:,i]!=X[:,j]).sum()
                  for i in range(X.shape[1]) for j in range(i+1,X.shape[1])
              ])) if X.shape[1]>1 else 0,
              avg_row_sum=float(X.sum(axis=1).mean()),
              row_sum_std=float(X.sum(axis=1).std()),
              improved_gradient_sensitivity=imp_g,
              effective_rank=compute_effective_rank(X),
              divergence_rate=div_rt,
              avg_tree_depth=td,
              Predicted_Error=float(np.nan),  # will fill in later if needed
              True_Error=err
            )
            for i,c in enumerate(founder_cols):
                rec[f"lsei_{c}"]        = float(lsei[i])
                rec[f"avg_SNPfreq_{c}"] = float(avgf[i])
            recs.append(rec)
        return pd.DataFrame.from_records(recs)


# ────────────────────────────────── main ─────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser(prog="haplomatic-window",
        description="Produce sliding-window feature tables with resume support.")
    ap.add_argument("--populations",  required=True,
                    help="one sim name per line")
    ap.add_argument("--founders-file",required=True,
                    help="one founder name per line")
    ap.add_argument("--snp-freqs",    required=True,
                    help="CSV with CHROM,pos,<founders>,<sims…>")
    ap.add_argument("--true-freq-dir",required=True,
                    help="dir with <pop>_true_freqs.csv")
    ap.add_argument("--window-sizes-kb",nargs="+",type=int,
                    default=[30,50,70,90,120,150,200,250])
    ap.add_argument("--stride-kb",    type=int, default=20)
    ap.add_argument("--min-snps",     type=int, default=10)
    ap.add_argument("--output",       required=True,
                    help="path to output CSV (appended to)")
    args = ap.parse_args()

    pops     = read_list_file(args.populations)
    founders = read_list_file(args.founders_file)
    snp_df   = pd.read_csv(args.snp_freqs)

    gen     = FixedWindowGenerator(tuple(args.window_sizes_kb),
                                   stride_kb=args.stride_kb,
                                   min_snps_per_window=args.min_snps)
    builder = FeatureBuilder()

    out_path       = Path(args.output)
    header_written = out_path.exists()

    # load already-done windows
    done = set()
    if header_written:
        prev = pd.read_csv(out_path, usecols=["sim","start","end"])
        done = set(zip(prev.sim, prev.start, prev.end))

    BATCH_SIZE = 2000
    buffer: List[pd.DataFrame] = []
    count = 0

    for pop in pops:
        print(f"[windows] processing {pop}", file=sys.stderr)
        tf = Path(args.true_freq_dir)/f"{pop}_true_freqs.csv"
        if not tf.exists():
            raise FileNotFoundError(tf)
        true_df = pd.read_csv(tf).loc[:,["pos",*founders]]

        obs_cols = ["CHROM","pos",*founders,pop]
        obs_df   = snp_df.loc[:,obs_cols].copy()

        windows = gen.generate(obs_df,true_df,founders,sim_col=pop)

        for key,info in windows.items():
            sim,(start,end) = key
            if (sim,start,end) in done:
                continue
            df_single = builder.build({key:info},founders,sim_col=pop)
            buffer.append(df_single)
            count += 1
            done.add((sim,start,end))

            if count >= BATCH_SIZE:
                pd.concat(buffer,ignore_index=True).to_csv(
                    out_path,
                    mode="a",
                    header=not header_written,
                    index=False
                )
                header_written = True
                buffer.clear()
                count = 0

    # flush any remainder
    if buffer:
        pd.concat(buffer,ignore_index=True).to_csv(
            out_path,
            mode="a",
            header=not header_written,
            index=False
        )

    print("Done. windows written to", out_path, file=sys.stderr)


if __name__=="__main__":
    main()
