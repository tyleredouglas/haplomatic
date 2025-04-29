# tests/test_read_simulator.py

import pandas as pd
import pytest
import os

from haplomatic.read_simulator import ReadSimulator

def test_generate_reads(tmp_path):
    # 1) Write a tiny FASTA with two haplotypes for one contig ("chrTest")
    fasta = tmp_path / "chrTest.fasta"
    fasta.write_text(
        """>B1
AAAAAAAAAA
>B2
CCCCCCCCCC
"""
    )

    # 2) Build a toy 'population' DataFrame:
    #    - MultiIndex index: contig level + position level
    #    - Two chromosome‐columns (c1, c2) carrying haplotype labels
    idx = pd.MultiIndex.from_product(
        [['chrTest'], [100, 200]], names=['contig','pos']
    )
    pop_df = pd.DataFrame({
        'c1': ['B1','B2'],
        'c2': ['B2','B1']
    }, index=idx)

    # 3) Instantiate ReadSimulator over that contig
    sim = ReadSimulator(
        fasta_paths={'chrTest': str(fasta)},
        regions=['chrTest']
    )

    # 4) Generate 10 paired‐end reads, writing to tmp_path/"reads_1.fastq" & "_2.fastq"
    out_prefix = str(tmp_path / "reads")
    hap_counts = sim.generate_reads(population=pop_df, n_reads=10, out_prefix=out_prefix)

    # 5a) The returned hap_counts must sum to exactly 10
    assert isinstance(hap_counts, pd.DataFrame)
    assert hap_counts.values.sum() == 10

    # 5b) The FASTQ files must exist
    assert os.path.exists(out_prefix + "_1.fastq")
    assert os.path.exists(out_prefix + "_2.fastq")

