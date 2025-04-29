# tests/test_pop_simulator.py

import pandas as pd
import numpy as np
import pytest

from haplomatic.pop_simulator import PopulationSimulator

def test_simulate_and_freqs_sum_to_one():
    # -- 1) Toy RIL DataFrame
    data = {
        'Founder1': [1, 0],
        'Founder2': [0, 1]
    }
    ril_df = pd.DataFrame(data, index=[100, 200])

    # -- 2) Simulate 2 diploid flies (4 haplotypes), 1 gen, no recomb
    sim = PopulationSimulator(
        ril_df=ril_df,
        n_flies=2,
        n_generations=1,
        recombination_rate=0.0
    )
    pop_df = sim.simulate()

    # Expect 2 rows × 4 columns
    assert pop_df.shape == (2, 4)

    # -- 3) Compute true frequencies
    freqs = sim.get_true_freqs(pop_df)

    # Should have 2 rows
    assert freqs.shape[0] == 2

    # Columns should be the distinct haplotype labels from ril_df (0 and 1)
    expected = set(ril_df.values.flatten())
    assert set(freqs.columns) == expected

    # -- 4) Each row must sum to 1.0
    row_sums = freqs.sum(axis=1).values
    assert np.allclose(row_sums, 1.0), f"Row sums were {row_sums}"

