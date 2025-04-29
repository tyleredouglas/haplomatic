# tests/test_trainer.py

import pandas as pd
import numpy as np
import pytest
import os
from pathlib import Path

from haplomatic.trainer import ModelTrainer

def make_toy_data(tmp_path):
    # 1) Write SNP freqs CSV: 2 positions, 2 haps, 1 sim
    snp_df = pd.DataFrame([
        ["chr1", 100, 1, 0, 0.6],
        ["chr1", 200, 0, 1, 0.4],
    ], columns=["chrom","pos","B1","B2","sim1"])
    snp_path = tmp_path/"snp_freqs.csv"
    snp_df.to_csv(snp_path, header=False, index=False)

    # 2) Write features CSV with exactly two windows (rows)
    cols = [
        "sim","chrom","start","end","window_bp","n_snps",
        "avg_uncertainty","avg_skew","avg_kurtosis","snr",
        "condition_number","min_col_distance","avg_row_sum","row_sum_std",
        "improved_gradient_sensitivity","effective_rank",
        "divergence_rate","avg_tree_depth",
        "lsei_B1","lsei_B2","avg_SNPfreq_B1","avg_SNPfreq_B2",
        "error"
    ]
    # Two toy windows on chr1
    feats = [
        ["sim1","chr1",100,200,100,2, 0.1,0.0,0.0,1.0, 1.0,1,1.0,0.0, 0.0,1.0, 0.0,1, 0.6,0.4,0.6,0.4, 0.2],
        ["sim1","chr1",200,300,100,2, 0.2,0.1,0.1,1.1, 1.2,1,1.0,0.0, 0.0,1.0, 0.0,1, 0.5,0.5,0.4,0.6, 0.1],
    ]
    feat_df = pd.DataFrame(feats, columns=cols)
    feat_path = tmp_path/"features.csv"
    feat_df.to_csv(feat_path, index=False)

    return str(snp_path), str(feat_path)

def test_load_data_splits(tmp_path):
    snp_path, feat_path = make_toy_data(tmp_path)

    # Trainer for region=chr1
    trainer = ModelTrainer(
        SNP_freqs_path = snp_path,
        features_path  = feat_path,
        hap_names      = ["B1","B2"],
        region         = "chr1",
        max_snps       = 10,
        batch_size     = 1,
        val_batch_size = 1,
        num_workers    = 0,
        epochs         = 0,   # we won't call train()
        lr             = 1e-3,
        weight_decay   = 0.0
    )
    trainer.load_data()

    # We had 2 windows; 70% train => 1, 30% val => 1
    assert len(trainer.train_ds) == 1
    assert len(trainer.val_ds)   == 1

    # Check that the DataLoader yields the correct shapes
    batch = next(iter(trainer.train_loader))
    X_raw, X_tab, y = batch
    # X_raw should be [batch_size, max_snps, n_haps] = [1,10,2]
    assert X_raw.shape == (1, trainer.max_snps, 2)
    # X_tab should be [1, n_tab_feats]; here n_tab_feats = total_feat_cols = 22 - 5 metadata = 17
    # but since batch_size=1 we just check dims
    assert X_tab.ndim == 2 and X_tab.shape[0] == 1
    # y should be [1]
    assert y.shape == (1,)

def test_region_filtering(tmp_path):
    snp_path, feat_path = make_toy_data(tmp_path)

    # Trainer for a region with no data
    trainer = ModelTrainer(
        SNP_freqs_path = snp_path,
        features_path  = feat_path,
        hap_names      = ["B1","B2"],
        region         = "chr2",  # no chr2 in our toy data
        epochs         = 0
    )
    trainer.load_data()

    # No windows should survive filtering
    assert len(trainer.train_ds) == 0
    assert len(trainer.val_ds)   == 0

