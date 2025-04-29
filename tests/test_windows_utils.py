# tests/test_windows_utils.py

import pandas as pd
import numpy as np
import pytest

from haplomatic.windows import (
    FixedWindowGenerator,
    lsei_haplotype_estimator,
    compute_avg_snpfreqs,
    compute_effective_rank
)

def test_lsei_estimator_identity():
    # X = I, so we should recover b exactly
    X = np.eye(3)
    b = np.array([0.2, 0.5, 0.3])
    p = lsei_haplotype_estimator(X, b)
    assert pytest.approx(p.tolist()) == [0.2, 0.5, 0.3]

def test_compute_avg_snpfreqs_basic():
    # B1 present in rows 0 & 2; B2 in rows 1 & 3
    df = pd.DataFrame({
        'B1': [1, 0, 1, 0],
        'B2': [0, 1, 0, 1],
        'sim': [0.1, 0.2, 0.3, 0.4]
    })
    avg = compute_avg_snpfreqs(df, 'sim', ['B1','B2'])
    # B1 avg = (0.1 + 0.3)/2 = 0.2; B2 avg = (0.2 + 0.4)/2 = 0.3
    assert pytest.approx(avg.tolist()) == [0.2, 0.3]

def test_compute_effective_rank_known():
    # 2×2 identity has two singular values of 1 => effective rank = 2
    X = np.eye(2)
    er = compute_effective_rank(X)
    assert pytest.approx(er, rel=1e-6) == 2.0

def test_fixed_window_generator_single_window():
    # Build an observed DataFrame with 3 SNPs spanning 1000 bp
    obs = pd.DataFrame({
        'chrom': ['chr1'] * 3,
        'pos':   [100, 600, 1100],
        'B1':    [1, 0, 1],
        'B2':    [0, 1, 0],
        'sim1':  [0.1, 0.2, 0.3]
    })
    # True frequencies at exactly those positions
    true = pd.DataFrame({
        'pos': [100, 600, 1100],
        'B1':  [0.1, 0.5, 0.9],
        'B2':  [0.9, 0.5, 0.1]
    })

    # Use a 1 kb window: anchor=100 → [100,1100)
    gen = FixedWindowGenerator(
        window_sizes_kb=(1,),   # 1 kb
        stride_kb=1,
        min_snps_per_window=1
    )
    windows = gen.generate(obs, true, 'sim1')

    # Expect exactly one window
    assert len(windows) == 1

    (sim_key, (start, end)), info = next(iter(windows.items()))
    assert sim_key == 'sim1'
    assert start == 100 and end == 1100

    # The window should include SNPs at pos 100 and 600 (1100 is excluded)
    wdf = info['window']
    assert list(wdf['pos']) == [100, 600]

    # median = (100+600)/2 = 350 → nearest true pos is 100 (tie goes to first)
    assert pytest.approx(info['true_freq_row'].tolist()) == [0.1, 0.9]

