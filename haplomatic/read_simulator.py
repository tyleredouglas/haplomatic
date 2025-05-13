# haplomatic/read_simulator.py
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple
from Bio import SeqIO

class ReadSimulator:
    """
    Simulate paired-end reads from a population DataFrame whose index is
    (CHROM, pos) and whose columns are founder IDs.

    Parameters
    ----------
    fasta_paths : Dict[str, str]
        Mapping contig → FASTA path (each FASTA contains all founder sequences).
    regions : List[str]
        Contigs to simulate over (must appear in population.index level 0).
    """

    # ------------------------------------------------------------------ #
    # constructor & helpers
    # ------------------------------------------------------------------ #
    def __init__(self,
                 fasta_paths: Dict[str, str],
                 regions: List[str]):
        self.regions = regions
        # load sequences only for requested contigs
        self.sequences: Dict[str, Dict[str, str]] = {
            contig: self._read_fasta(fp)
            for contig, fp in fasta_paths.items()
            if contig in regions
        }
        # collect haplotype IDs
        hap_ids = {hap for d in self.sequences.values() for hap in d}
        self.hap_names = sorted(hap_ids)

    @staticmethod
    def _read_fasta(fp: str) -> Dict[str, str]:
        """Return {record.id: sequence} from FASTA."""
        return {rec.id: str(rec.seq) for rec in SeqIO.parse(fp, "fasta")}

    @staticmethod
    def reverse_complement(seq: str) -> str:
        comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
        return "".join(comp.get(b, b) for b in seq)

    # ------------------------------------------------------------------ #
    # choose coordinates for one read-pair
    # ------------------------------------------------------------------ #
    def _choose_coordinates(
        self,
        population: pd.DataFrame
    ) -> Tuple[str, int, int, int, int, int, int]:
        # draw lengths
        lf   = int(round(np.random.normal(150, 10)))
        lr   = int(round(np.random.normal(150, 10)))
        frag = int(round(np.random.normal(500, 50)))
        # gap between forward and reverse
        gap  = frag - (lf + lr)

        # pick contig and random forward start
        contig = random.choice(self.regions)
        chrom_df = population.loc[contig]
        pos_vals = chrom_df.index.get_level_values("pos")
        start_f = random.randint(int(pos_vals.min()), int(pos_vals.max()))
        end_f   = start_f + lf

        # compute reverse-read coordinates *after* the gap
        start_r = end_f + gap
        end_r   = start_r + lr

        return contig, start_f, end_f, lf, start_r, end_r, lr

    # ------------------------------------------------------------------ #
    # main generator
    # ------------------------------------------------------------------ #
    def generate_reads(
        self,
        population: pd.DataFrame,
        n_reads: int,
        out_prefix: str
    ) -> pd.DataFrame:
        """
        Produce n_reads paired-end reads, write:
          {out_prefix}_1.fastq / _2.fastq
        and return a depth table indexed by (contig,pos).
        """
        # initialize depth counter
        hap_counts = pd.DataFrame(
            0,
            index=population.index,
            columns=self.hap_names
        )

        with open(f"{out_prefix}_1.fastq", "w") as fq1, \
             open(f"{out_prefix}_2.fastq", "w") as fq2:

            read_cnt = 0
            while read_cnt < n_reads:
                # unpack coordinates
                contig, sf, ef, lf, sr, er, lr = self._choose_coordinates(population)
                chrom_df = population.loc[contig]

                # pick a random haplotype column
                hap_col  = random.choice(list(chrom_df.columns))
                template = chrom_df[hap_col]

                # find nearest SNP to forward-read start
                pos_array  = template.index.to_numpy()
                nearest_pos = int(pos_array[np.argmin(np.abs(pos_array - sf))])
                hap = template.loc[nearest_pos]
                if isinstance(hap, pd.Series):  # handle duplicate positions
                    hap = hap.iloc[0]

                # increment depth at the SNP
                hap_counts.at[(contig, nearest_pos), hap] += 1

                # generate read sequences
                fwd_seq = self.sequences[contig][hap][sf:ef]
                rev_seq = self.reverse_complement(
                    self.sequences[contig][hap][sr:er]
                )[::-1]

                read_id = random.randint(1000, 9999)
                fq1.write(f"@{contig}_{sf}_{read_id}/1\n{fwd_seq}\n+\n{'I'*len(fwd_seq)}\n")
                fq2.write(f"@{contig}_{sf}_{read_id}/2\n{rev_seq}\n+\n{'I'*len(rev_seq)}\n")

                read_cnt += 1

        return hap_counts
