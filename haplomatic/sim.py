#!/usr/bin/env python3
from __future__ import annotations
import os
import math
import argparse
import textwrap
import pathlib
import re
import shutil
import concurrent.futures
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from .pop_simulator import simulate_population, get_true_freqs
from .read_simulator import ReadSimulator

# ──────────────────────────────────────────────────────────────────────────────
# required helper utilities
# ──────────────────────────────────────────────────────────────────────────────
def read_hap_file(path: str) -> List[str]:
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]

def read_regions_file(path: str) -> List[Tuple[str, int, int]]:
    out = []
    with open(path) as fh:
        for raw in fh:
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue
            if re.match(r"^[^:\s]+:\d+:\d+$", ln):         
                chrom, start, end = ln.split(":")
            else:                                          
                chrom, start, end, *_ = ln.split()
            out.append((chrom, int(start), int(end)))
    return out

def simulate_rils_blocks(
    founder_ids: List[str],
    regions: List[Tuple[str, int, int]],
    n_rils: int,
    step_bp: int       = 10_000,
    mean_block_bp: int = 100_000,
    k_active: int      = 3,
    alpha_base: float  = 0.25,
    seed: int | None   = 42,
) -> pd.DataFrame:
    rng          = np.random.default_rng(seed)
    ril_cols     = [f"RIL{i}" for i in range(1, n_rils + 1)]
    cur_fndr     = {ril: None for ril in ril_cols}
    bp_left      = {ril: 0    for ril in ril_cols}
    records: list[dict] = []
    for chrom, start, end in regions:
        for pos in range(start, end + 1, step_bp):
            active   = rng.choice(founder_ids, size=k_active, replace=False)
            weights  = rng.dirichlet([alpha_base] * k_active)
            row = {"CHROM": chrom, "pos": pos}
            for ril in ril_cols:
                if bp_left[ril] <= 0:
                    cur_fndr[ril] = rng.choice(active, p=weights)
                    bp_left[ril]  = max(int(rng.exponential(mean_block_bp)), step_bp)
                row[ril] = cur_fndr[ril]
                bp_left[ril] -= step_bp
            records.append(row)
    return pd.DataFrame.from_records(records)


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            simulate populations and paired-end reads.

            required files
            --------------
              --haplotypes  list.txt        one founder ID per line
              --regions     regions.bed     contig:start:end

            RIL table
            ----------
              • provide one via --ril-df   (CHROM,pos,RIL1..)
              • OR simulate it with --n-rils etc.

            """
        ),
    )

    # always-required
    p.add_argument("--haplotypes", required=True,
                   help="text file of haplotype IDs")
    p.add_argument("--regions",    required=True,
                   help="text file of target regions (chrom start end)")

    # RIL table options
    p.add_argument("--ril-df",
                   help="CSV, first two columns are contig and position, remaining columns are RILs")
    p.add_argument("--n-rils",     type=int, default=300,
                   help="If no --ril-df, number of synthetic RILs to create")
    p.add_argument("--step-bp",    type=int, default=10_000,
                   help="Marker spacing for synthetic RIL genotype values")
    p.add_argument("--mean-block", type=int, default=100_000,
                   help="mean haplotype block length for synthetic RILs(bp)")
    p.add_argument("--k-active",   type=int, default=3,
                   help="haplotypes allowed per locus in RILs")
    p.add_argument("--alpha-base", type=float, default=0.25,
                   help="dirichlet for haplotype freq sampling")

    # population evolution + read parameters
    p.add_argument("--n-flies",     type=int, default=300)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--recomb-rate", type=float, default=0.5)
    p.add_argument("--n-sims",      type=int, default=2)

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--coverage",  type=float,
                     help="target coverage across all regions")
    grp.add_argument("--read-depth", type=int,
                     help="explicit read-pair count (alternative to coverage)")

    # allow user to adjust parallelism
    p.add_argument(
        "--threads", "--n-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="number of parallel worker threads for read generation"
    )
    p.add_argument("--read-length",   type=int, default=150)
    p.add_argument("--founder-fastas", nargs="+", required=True,
                   help="one FASTA per contig (seq IDs = haplotypes)")
    p.add_argument("--contigs", nargs="+", required=True,
                   help="contig names (must match names in RIL df)")
    p.add_argument("--output-dir", default=".",
                   help="save results")

    args = p.parse_args()
    num_workers = args.threads
    print(f"Using {num_workers} read‐generation threads")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    founder_ids = read_hap_file(args.haplotypes)
    regions     = read_regions_file(args.regions)

    # --- obtain a wide RIL dataframe ----------------------------------------
    if args.ril_df:
        ril_df = pd.read_csv(args.ril_df)
    else:
        ril_df = simulate_rils_blocks(
            founder_ids   = founder_ids,
            regions       = regions,
            n_rils        = args.n_rils,
            step_bp       = args.step_bp,
            mean_block_bp = args.mean_block,
            k_active      = args.k_active,
            alpha_base    = args.alpha_base,
        )
        ril_df.to_csv(out_dir / "seed_RILs.csv", index=False)
        print(f"outputting synthetic RILs to {out_dir/'seed_rils.csv'}")

    ril_wide = ril_df.set_index(["CHROM", "pos"]).sort_index()

    # --- total bp across all requested regions ------------------------------
    total_bp = sum(end - start + 1 for _, start, end in regions)
    if args.coverage is not None:
        bases_per_pair = 2 * args.read_length
        args.read_depth = math.ceil(args.coverage * total_bp / bases_per_pair)
        print(f"Target {args.coverage}× ≈ {args.read_depth:,} read-pairs")
    else:
        print(f"Fixed {args.read_depth:,} read-pairs")

    # --- set up read simulator ----------------------------------------------
    fasta_map: Dict[str, str] = dict(zip(args.contigs, args.founder_fastas))
    rsim = ReadSimulator(fasta_paths=fasta_map, regions=args.contigs)

    # --- run replicate simulations -----------------------------------------
    for i in range(1, args.n_sims + 1):
        tag        = f"sim{i}"
        pop_path   = out_dir / f"{tag}_pop.csv"
        true_path  = out_dir / f"{tag}_true_freqs.csv"
        depth_path = out_dir / f"{tag}_depth.csv"
        fq1_path   = out_dir / f"{tag}_1.fastq"
        fq2_path   = out_dir / f"{tag}_2.fastq"

        # resume if everything is already there
        if pop_path.exists() and true_path.exists() and depth_path.exists() \
           and fq1_path.exists() and fq2_path.exists():
            print(f"Skipping {tag}, outputs already exist.")
            continue

        print(f"\n— {tag} —")

        # 1) simulate population & true freqs
        pop_df  = simulate_population(
            RIL_matrix        = ril_wide,
            n_flies           = args.n_flies,
            n_generations     = args.generations,
            recombination_rate= args.recomb_rate,
        )
        pop_df.to_csv(pop_path)

        true_df = get_true_freqs(pop_df)
        true_df.to_csv(true_path)

        # 2) parallel read generation
        base_chunk = args.read_depth // num_workers
        sizes = [
            base_chunk + (1 if j < args.read_depth % num_workers else 0)
            for j in range(num_workers)
        ]

        def _gen_chunk(chunk_reads: int, part_prefix: pathlib.Path) -> pd.DataFrame:
            return rsim.generate_reads(pop_df, chunk_reads, str(part_prefix))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as exe:
            futures = [exe.submit(_gen_chunk, sz, out_dir / f"{tag}_part{j}")
                       for j, sz in enumerate(sizes)]
            parts = [f.result() for f in futures]

        # concatenate FASTQs
        with open(fq1_path, "wb") as out1, open(fq2_path, "wb") as out2:
            for j in range(len(parts)):
                p1 = out_dir / f"{tag}_part{j}_1.fastq"
                p2 = out_dir / f"{tag}_part{j}_2.fastq"
                with open(p1, "rb") as in1:
                    shutil.copyfileobj(in1, out1)
                with open(p2, "rb") as in2:
                    shutil.copyfileobj(in2, out2)
                p1.unlink()
                p2.unlink()

        # sum up depth tables
        depth_df = parts[0]
        for df_chunk in parts[1:]:
            depth_df = depth_df.add(df_chunk, fill_value=0).astype(int)
        depth_df.to_csv(depth_path)

        empirical_cov = depth_df.values.sum() * args.read_length * 2 / total_bp
        print(f"read-pairs         : {args.read_depth:,}")
        print(f"coverage           : {empirical_cov:,.2f}×")

    print("\nsimulations complete.")

if __name__ == "__main__":
    main()
