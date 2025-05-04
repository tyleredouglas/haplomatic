#!/usr/bin/env python3
# haplomatic/trainer.py
# -------------------------------------------------------------
# Train / resume the SNP-window error-predictor transformer
# -------------------------------------------------------------
import os
import sys
import argparse
import math

import numpy  as np
import pandas as pd
import numpy._core.multiarray as marray
from torch.serialization import add_safe_globals

from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.utils.data    import Dataset, DataLoader
from   torch.cuda.amp      import autocast, GradScaler


def read_list(txt: str):
    with open(txt) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


# ───────────────────────── Dataset ────────────────────────────
class ErrorDataset(Dataset):
    """Return (X_raw, X_tab, y) for one window."""
    def __init__(self, df_win: pd.DataFrame, snp_df: pd.DataFrame,
                 idx: np.ndarray, X_tab: np.ndarray, y: np.ndarray,
                 max_snps: int, hap_cols: list[str]):

        self.df   = df_win.loc[idx].reset_index(drop=True)
        self.snp  = snp_df.sort_values('pos').reset_index(drop=True)
        self.pos  = self.snp['pos'].values
        self.haps = hap_cols
        self.max  = max_snps
        self.Xtab = torch.from_numpy(X_tab[idx]).float()
        self.y    = torch.from_numpy(y[idx]).float()

        self.ranges = []
        for _, row in self.df.iterrows():
            lo = np.searchsorted(self.pos, row.start, side='left')
            hi = np.searchsorted(self.pos, row.end,   side='left')
            self.ranges.append((lo, hi))

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        lo, hi = self.ranges[i]
        sub    = self.snp.iloc[lo:hi][self.haps].astype(float).values
        n, k   = sub.shape
        if n >= self.max:
            mat = sub[-self.max:]
        else:
            pad = np.zeros((self.max-n, k), dtype=np.float32)
            mat = np.vstack([pad, sub])

        return (
            torch.from_numpy(mat.astype(np.float32)),
            self.Xtab[i],
            self.y[i]
        )


# ───────────────────────── Network ────────────────────────────
class SNPTransformerEncoder(nn.Module):
    def __init__(self, n_haps:int, embed:int=64, heads:int=4, layers:int=3):
        super().__init__()
        self.proj  = nn.Linear(n_haps, embed)
        block      = nn.TransformerEncoderLayer(embed, heads, batch_first=True)
        self.enc   = nn.TransformerEncoder(block, layers)
        self.pool  = nn.Linear(embed, 1)

    def forward(self, x):
        z = self.proj(x)
        z = self.enc(z)
        w = F.softmax(self.pool(z), dim=1)
        return (w * z).sum(dim=1)


class TabularRegressor(nn.Module):
    def __init__(self, inp:int, hidden=(256,128), drop=0.3):
        super().__init__()
        layers, dims = [], [inp] + list(hidden)
        for i in range(len(hidden)):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(drop)
            ]
        layers.append(nn.Linear(dims[-1],1))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x).squeeze(1)


class Predictor(nn.Module):
    def __init__(self, n_haps:int, n_tab:int, embed:int=64, dropout:float=0.3):
        super().__init__()
        self.enc = SNPTransformerEncoder(n_haps, embed)
        self.reg = TabularRegressor(embed + n_tab, drop=dropout)

    def forward(self, Xraw, Xtab):
        z = self.enc(Xraw)
        return self.reg(torch.cat([z, Xtab], dim=1))


# ──────────────────────── training fn ─────────────────────────
def train(model, train_loader, val_loader,
          opt, sched, scaler_grad, device,
          start_epoch, n_epochs, ckpt_path, best_ckpt_path,
          scaler_feat, best_r2, history,
          save_best, region_label="ALL"):

    print(f"[trainer] starting training: "
          f"{len(train_loader.dataset):,} windows "
          f"(val {len(val_loader.dataset):,}) | epochs {n_epochs}",
          flush=True)

    for epoch in range(start_epoch, n_epochs+1):
        # ---- training pass ----
        model.train()
        tot = 0.0
        for Xr,Xt,y in train_loader:
            Xr,Xt,y = Xr.to(device), Xt.to(device), y.to(device)
            opt.zero_grad()
            with autocast():
                pred = model(Xr,Xt)
                loss = F.mse_loss(pred, y)
            scaler_grad.scale(loss).backward()
            scaler_grad.step(opt)
            scaler_grad.update()
            tot += loss.item() * Xr.size(0)
        train_mse = tot / len(train_loader.dataset)

        # ---- validation pass ----
        model.eval()
        vtot, preds, trues = 0.0, [], []
        with torch.no_grad():
            for Xr,Xt,y in val_loader:
                Xr,Xt,y = Xr.to(device), Xt.to(device), y.to(device)
                with autocast():
                    pred = model(Xr,Xt)
                vtot += F.mse_loss(pred, y).item() * Xr.size(0)
                preds.append(pred.cpu().numpy())
                trues.append(y.cpu().numpy())
        val_mse = vtot / len(val_loader.dataset)

        # concatenate & drop any non-finite pairs before R²
        y_true = np.concatenate(trues)
        y_pred = np.concatenate(preds)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if not finite.all():
            print(f"[trainer] ⚠️  dropping {len(finite)-finite.sum()} NaN/Inf predictions", file=sys.stderr)
            y_true = y_true[finite]
            y_pred = y_pred[finite]

        val_r2  = r2_score(y_true, y_pred)
        sched.step(val_mse)

        # optionally save best
        if save_best and val_r2 > best_r2:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "opt_state_dict":   opt.state_dict(),
                "gscaler":          scaler_grad.state_dict(),
                "scaler_mean":      scaler_feat.mean_,
                "scaler_scale":     scaler_feat.scale_,
                "best_r2":          val_r2,
                "history":          history + [(epoch, train_mse, val_mse, val_r2)]
            }, best_ckpt_path)
            print(f"[new best R²={val_r2:.3f}, saved to {best_ckpt_path}")

        # record & rolling checkpoint
        history.append((epoch, train_mse, val_mse, val_r2))
        best_r2 = max(best_r2, val_r2)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "opt_state_dict":   opt.state_dict(),
            "gscaler":          scaler_grad.state_dict(),
            "scaler_mean":      scaler_feat.mean_,
            "scaler_scale":     scaler_feat.scale_,
            "best_r2":          best_r2,
            "history":          history
        }, ckpt_path)

        print(f"[{region_label}] epoch {epoch:02d} "
              f"train MSE={train_mse:.4e} val MSE={val_mse:.4e} R²={val_r2:.3f}",
              flush=True)

    return best_r2, history


# ─────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Train / resume transformer error-predictor"
    )
    ap.add_argument("--snp-freqs-csv",  required=True)
    ap.add_argument("--features-csv",   required=True)
    ap.add_argument("--feature-list",   required=True)
    ap.add_argument("--hap-names-file", required=True)
    ap.add_argument("--region",         default=None)
    ap.add_argument("--max-snps",       type=int,   default=400)
    ap.add_argument("--batch",          type=int,   default=64)
    ap.add_argument("--val-batch",      type=int,   default=128)
    ap.add_argument("--epochs",         type=int,   default=40)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--weight-decay",   type=float, default=1e-5)
    ap.add_argument("--workers",        type=int,   default=4)
    ap.add_argument("--dropout",        type=float, default=0.3,
                    help="dropout rate for the TabularRegressor")
    ap.add_argument("--model-name",     default=None,
                    help="root name for checkpoint and final model files")
    ap.add_argument("--save-best",      action="store_true",
                    help="also save the best model seen so far")
    args = ap.parse_args()

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}", file=sys.stderr)

    # read feature & haplotype column lists
    feat_cols = read_list(args.feature_list)
    hap_cols  = read_list(args.hap_names_file)

    # load your features CSV
    feat_df = pd.read_csv(args.features_csv)
    if args.region:
        feat_df = feat_df.query("chrom == @args.region").reset_index(drop=True)

    # 1) check for missing columns
    missing = set(feat_cols) - set(feat_df.columns)
    if missing:
        print(f"Missing feature columns in {args.features_csv}: {missing}", file=sys.stderr)
        sys.exit(1)

    # 2) pull out feature matrix & replace infinities
    X_tab_df = feat_df[feat_cols].replace([np.inf, -np.inf], np.nan)
    before   = len(feat_df)
    mask     = ~X_tab_df.isna().any(axis=1)
    feat_df  = feat_df.loc[mask].reset_index(drop=True)
    X_tab    = X_tab_df.loc[mask].values
    after    = len(feat_df)
    print(f"[trainer] dropped {before-after} rows with NaN/Inf in features", file=sys.stderr)

    # 3) sanity check: no NaNs left
    if np.isnan(X_tab).any():
        print(f"NaNs still found after cleaning—aborting", file=sys.stderr)
        sys.exit(1)

    # target
    y_all = feat_df["error"].clip(lower=1e-8).values

    # load SNP‐freqs
    snp_df = pd.read_csv(args.snp_freqs_csv)
    if args.region:
        snp_df = snp_df.query("chrom == @args.region").reset_index(drop=True)

    # split & scale
    idx = np.arange(len(feat_df))
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(0.7 * len(idx))
    tr_idx, val_idx = idx[:split], idx[split:]
    scaler = StandardScaler().fit(X_tab[tr_idx])
    X_scaled = scaler.transform(X_tab)

    # build datasets & loaders
    train_ds = ErrorDataset(feat_df, snp_df, tr_idx, X_scaled, y_all,
                            args.max_snps, hap_cols)
    val_ds   = ErrorDataset(feat_df, snp_df, val_idx, X_scaled, y_all,
                            args.max_snps, hap_cols)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=True,
                              persistent_workers=(args.workers>0))
    val_loader   = DataLoader(val_ds,   batch_size=args.val_batch,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True,
                              persistent_workers=(args.workers>0))

    # model & optimizer setup
    region_label = args.region or "ALL"
    model_name   = args.model_name or f"predictor_{region_label}"
    ckpt_path    = f"{model_name}.pt"
    best_ckpt    = f"{model_name}_best.pt"
    model        = Predictor(len(hap_cols),
                             X_scaled.shape[1],
                             dropout=args.dropout
                            ).to(device)

    opt    = optim.Adam(model.parameters(),
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    sched  = optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                 'min',
                                                 patience=4)
    gscaler= GradScaler()

    # resume state if checkpoint exists
    start_epoch, best_r2, history = 1, -math.inf, []
    if os.path.exists(ckpt_path):
        add_safe_globals([marray._reconstruct])
        ck = torch.load(ckpt_path,
                        map_location=device,
                        weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["opt_state_dict"])
        gscaler.load_state_dict(ck["gscaler"])
        scaler.mean_, scaler.scale_ = ck["scaler_mean"], ck["scaler_scale"]
        start_epoch = ck["epoch"] + 1
        best_r2     = ck["best_r2"]
        history     = ck["history"]
        print(f"[trainer] resumed from {ckpt_path} (epoch {start_epoch})",
              file=sys.stderr)

    # run training loop
    best_r2, history = train(
        model, train_loader, val_loader,
        opt, sched, gscaler, device,
        start_epoch, args.epochs,
        ckpt_path, best_ckpt,
        scaler, best_r2, history,
        save_best=args.save_best,
        region_label=region_label
    )

    # save final artifacts
    pd.DataFrame(history,
                 columns=["epoch","train_mse","val_mse","val_r2"]
                ).to_csv(f"training_history_{model_name}.csv",
                         index=False)
    torch.save(model.state_dict(),
               f"{model_name}.pth")

    print("training complete", file=sys.stderr)


if __name__ == "__main__":
    main()
