# haplomatic/pop_simulator.py

import pandas as pd
import random
from typing import Tuple

def recombine(
    chrom1: pd.Series,
    chrom2: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Given two haplotype‐label series (chrom1, chrom2),
    choose a random crossover point and return two offspring.
    """
    # pick a random crossover index
    point = random.randrange(0, len(chrom1))
    # offspring swap tails at that point
    offspring1 = pd.concat([chrom1.iloc[:point], chrom2.iloc[point:]])
    offspring2 = pd.concat([chrom2.iloc[:point], chrom1.iloc[point:]])
    return offspring1, offspring2

def simulate_population(
    RIL_matrix: pd.DataFrame,
    n_flies: int,
    n_generations: int,
    recombination_rate: float
) -> pd.DataFrame:
    """
    Simulate n_flies diploids (2*n_flies haplotypes) for n_generations
    given an initial RIL_matrix (index=(chrom,pos), columns=founders).
    Returns a DataFrame with the same index and 2*n_flies columns,
    each entry is a founder label.
    """
    # start by sampling 2*n_flies founder haplotypes at each position
    population = RIL_matrix.sample(n=n_flies*2, axis=1, replace=True)

    for _ in range(n_generations):
        new_pop = pd.DataFrame(index=population.index,
                               columns=population.columns)
        cols = list(population.columns)
        i = 0
        while i < len(cols):
            p1 = population.iloc[:, random.randrange(len(cols))]
            p2 = population.iloc[:, random.randrange(len(cols))]
            if random.random() < recombination_rate:
                c1, c2 = recombine(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_pop.iloc[:, i] = c1
            if i + 1 < len(cols):
                new_pop.iloc[:, i+1] = c2
            i += 2
        # drift: resample with replacement for next generation
        population = new_pop.sample(n=n_flies*2, axis=1, replace=True)

    return population

def get_true_freqs(simulated_pop: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of simulated haplotype labels (columns = hap copies),
    compute at each row the frequency of each founder label.
    Returns a DataFrame (same index) with one column per founder,
    containing the relative frequency at that position.
    """
    # count each label at every position
    counts = simulated_pop.apply(lambda row: row.value_counts(), axis=1).fillna(0)
    # convert to frequencies
    freqs = counts.div(counts.sum(axis=1), axis=0)
    return freqs
