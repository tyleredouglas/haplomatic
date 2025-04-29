# tests/test_windows.py

import pandas as pd
import numpy as np
import pytest

from haplomatic.windows import FixedWindowGenerator, FeatureBuilder

def test_fixed_window_generator():
    # -- observed SNP table (4 positions spanning ≥1 kb, 2 founders, 1 sim column)
    df_obs = pd.DataFrame({
        'chrom': ['chr1'] * 4,
        'pos':   [100, 500, 900, 1300],  # spans 1200 bp
        'B1':    [1, 0, 1, 0],
        'B2':    [0, 1, 0, 1],
        'sim1':  [0.5, 0.6, 0.7, 0.8]
    })

    # -- true frequencies at those positions
    df_true = pd.DataFrame({
        'pos': [100, 500, 900, 1300],
        'B1':  [0.5, 0.4, 0.5, 0.4],
        'B2':  [0.5, 0.6, 0.5, 0.6]
    })

    # Use 1 kb windows: start=100, end=1100
    gen = FixedWindowGenerator(window_sizes_kb=(1,), stride_kb=1, min_snps_per_window=2)
    windows = gen.generate(df_obs, df_true, 'sim1')

    # We expect exactly one window
    assert len(windows) == 1
    (sim_key, (start, end)), info = next(iter(windows.items()))
    assert sim_key == 'sim1'
    assert start == 100 and end == 1100

    # That window should include SNPs with pos ≥100 & <1100 → [100, 500, 900]
    wdf = info['window']
    assert list(wdf['pos']) == [100, 500, 900]

    # Median of [100,500,900] is 500 → true_freq_row from pos=500
    np.testing.assert_allclose(info['true_freq_row'], [0.4, 0.6])
    assert info['window_size_bp'] == 1000


def test_feature_builder_lsei_and_error(monkeypatch):
    # Create a single window with two SNPs, two founders
    wdf = pd.DataFrame({
        'B1':   [1, 0],
        'B2':   [0, 1],
        'sim1': [0.6, 0.4]
    })
    windows = {
        ('sim1', (100, 200)): {
            'window':         wdf,
            'true_freq_row':  np.array([0.5, 0.5]),
            'window_size_bp': 100
        }
    }

    # Build features but override MCMC to avoid heavy sampling
    fb = FeatureBuilder(num_warmup=1, num_samples=1, num_chains=1, rng_seed=0)
    monkeypatch.setattr(fb, '_run_mcmc', lambda X, b: (0.0, 0))  # no divergence, zero depth

    df_feat = fb.build(windows, founder_cols=['B1', 'B2'], sim_col='sim1')

    # Should produce exactly one row
    assert df_feat.shape[0] == 1

    row = df_feat.iloc[0]
    # LS‐estimate should equal [0.6,0.4] since X=[[1,0],[0,1]] minimizes error at p=b
    assert pytest.approx([row['lsei_B1'], row['lsei_B2']]) == [0.6, 0.4]

    # Average SNP‐freq per haplotype is just the mean of sim1 where B_i==1
    assert pytest.approx([row['avg_SNPfreq_B1'], row['avg_SNPfreq_B2']]) == [0.6, 0.4]

    # Error = |p_mean - true_freq| = |[0.6,0.4] - [0.5,0.5]| = [0.1,0.1] sum → 0.2
    assert pytest.approx(row['error'], rel=1e-6) == 0.2

    # MCMC diagnostics (patched) should appear
    assert row['divergence_rate'] == 0.0
    assert row['avg_tree_depth'] == 0

