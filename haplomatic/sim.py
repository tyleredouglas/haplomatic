#!/usr/bin/env python3
from __future__ import annotations
import os
import math
import argparse
import textwrap
import pathlib
import re
import shutil
import sys
import concurrent.futures
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from .pop_simulator import simulate_population, get_true_freqs
from .read_simulator import ReadSimulator


def read_hap_file(path: str) -> List[str]:
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def read_regions_file(path: str) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str,int,int]] = []
    with open(path) as fh:
        for raw in fh:
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue

            # chrom:start:end
            m = re.match(r"^([^:\s]+):(\d+):(\d+)$", ln)
            if m:
                chrom, start, end = m.groups()
            else:
                # chrom:start-end
                m2 = re.match(r"^([^:\s]+):(\d+)-(\d+)$", ln)
                if m2:
                    chrom, start, end = m2.groups()
                else:
                    # whitespace-separated
                    parts = ln.split()
                    if len(parts) != 3:
                        raise ValueError(f"Bad region spec: {ln}")
                    chrom, start, end = parts

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
    rng      = np.random.default_rng(seed)
    ril_cols = [f"RIL{i}" for i in range(1, n_rils+1)]
    cur      = {r: None for r in ril_cols}
    left     = {r: 0    for r in ril_cols}
    recs: list[dict] = []

    for chrom, start, end in regions:
        for pos in range(start, end+1, step_bp):
            active  = rng.choice(founder_ids, size=k_active, replace=False)
            weights = rng.dirichlet([alpha_base]*k_active)
            row = {"CHROM": chrom, "pos": pos}
            for r in ril_cols:
                if left[r] <= 0:
                    cur[r]  = rng.choice(active, p=weights)
                    left[r] = max(int(rng.exponential(mean_block_bp)), step_bp)
                row[r] = cur[r]
                left[r] -= step_bp
            recs.append(row)

    return pd.DataFrame.from_records(recs)


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            simulate populations and paired-end reads.

            required files
            --------------
              --haplotypes  list.txt        one founder ID per line
              --regions     regions.txt     contig:start:end (or chrom start end)
        """)
    )

    p.add_argument("--haplotypes", required=True,
                   help="text file of haplotype IDs")
    p.add_argument("--regions",    required=True,
                   help="text file of target regions")
    p.add_argument("--ril-df",
                   help="CSV of existing RIL table (CHROM,pos,RIL1..)")
    p.add_argument("--n-rils",     type=int,   default=300)
    p.add_argument("--step-bp",    type=int,   default=10_000)
    p.add_argument("--mean-block", type=int,   default=100_000)
    p.add_argument("--k-active",   type=int,   default=3)
    p.add_argument("--alpha-base", type=float, default=0.25)
    p.add_argument("--n-flies",     type=int,   default=300)
    p.add_argument("--generations", type=int,   default=20)
    p.add_argument("--recomb-rate", type=float, default=0.5)
    p.add_argument("--n-sims",      type=int,   default=2)

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--coverage",  type=float,
                     help="target coverage across all regions")
    grp.add_argument("--read-depth", type=int,
                     help="explicit read-pair count (alternative to coverage)")

    p.add_argument("--read-length",   type=int, default=150)
    p.add_argument("--founder-fastas", nargs="+", required=True,
                   help="one FASTA per contig (seq IDs = haplotypes)")
    p.add_argument("--contigs", nargs="+", required=True,
                   help="contig names (must match RIL df)")
    p.add_argument("--output-dir", default=".",
                   help="save results here")
    p.add_argument("--threads", "--n-workers",
                   type=int, default=os.cpu_count() or 1,
                   help="parallel read‐generation threads")

    args = p.parse_args()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load hap IDs & regions
    founders = read_hap_file(args.haplotypes)
    regions  = read_regions_file(args.regions)

    # build or load RIL table
    if args.ril_df:
        ril_df = pd.read_csv(args.ril_df)
    else:
        ril_df = simulate_rils_blocks(
            founder_ids   = founders,
            regions       = regions,
            n_rils        = args.n_rils,
            step_bp       = args.step_bp,
            mean_block_bp = args.mean_block,
            k_active      = args.k_active,
            alpha_base    = args.alpha_base,
        )
        ril_df.to_csv(out_dir/"seed_RILs.csv", index=False)
        print(f"wrote synthetic RILs to {out_dir/'seed_RILs.csv'}", file=sys.stderr)

    ril_wide = ril_df.set_index(["CHROM","pos"]).sort_index()

    # compute read depth
    total_bp = sum(e - s + 1 for _, s, e in regions)
    if args.coverage is not None:
        bp_per_pair = 2 * args.read_length
        args.read_depth = math.ceil(args.coverage * total_bp / bp_per_pair)
        print(f"Target {args.coverage}× ≈ {args.read_depth:,} read-pairs", file=sys.stderr)
    else:
        print(f"Fixed {args.read_depth:,} read-pairs", file=sys.stderr)

    # set up simulator
    fasta_map = dict(zip(args.contigs, args.founder_fastas))
    rsim      = ReadSimulator(fasta_paths=fasta_map, regions=args.contigs)

    # run sims
    for i in range(1, args.n_sims+1):
        tag       = f"sim{i}"
        pop_path  = out_dir/f"{tag}_pop.csv"
        true_path = out_dir/f"{tag}_true_freqs.csv"
        depth_path= out_dir/f"{tag}_depth.csv"
        fq1_path  = out_dir/f"{tag}_1.fastq"
        fq2_path  = out_dir/f"{tag}_2.fastq"

        # skip if done
        if (pop_path.exists() and true_path.exists() and
            depth_path.exists() and fq1_path.exists() and fq2_path.exists()):
            print(f"Skipping {tag}, outputs exist.", file=sys.stderr)
            continue

        print(f"\n— {tag} —", file=sys.stderr)

        # 1) simulate pop & true freqs
        pop_df  = simulate_population(
            RIL_matrix        = ril_wide,
            n_flies           = args.n_flies,
            n_generations     = args.generations,
            recombination_rate= args.recomb_rate,
        )
        pop_df.to_csv(pop_path)
        tf = get_true_freqs(pop_df)
        tf.to_csv(true_path)

        # 2) parallel read generation
        num_workers = min(args.threads, args.read_depth)
        base = args.read_depth // num_workers
        rem  = args.read_depth % num_workers
        sizes = [base + (j < rem) for j in range(num_workers)]

        parts: List[pd.DataFrame] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as exe:
            futures = [
                exe.submit(rsim.generate_reads, pop_df, sz, out_dir/f"{tag}_part{j}")
                for j, sz in enumerate(sizes)
            ]
            for f in futures:
                parts.append(f.result())

        # 3) sanity-check partial FASTQs
        for j in range(num_workers):
            p1 = out_dir/f"{tag}_part{j}_1.fastq"
            p2 = out_dir/f"{tag}_part{j}_2.fastq"
            cnt1 = sum(1 for _ in open(p1)) // 4
            cnt2 = sum(1 for _ in open(p2)) // 4
            print(f"Part {j} reads: +={cnt1}, -={cnt2}", file=sys.stderr)
            if cnt1 != cnt2:
                raise RuntimeError(f"Mismatch in part {j}: {cnt1} vs {cnt2}")

        # 4) stitch FASTQs
        with open(fq1_path, "wb") as out1, open(fq2_path, "wb") as out2:
            for j in range(num_workers):
                p1 = out_dir/f"{tag}_part{j}_1.fastq"
                p2 = out_dir/f"{tag}_part{j}_2.fastq"
                with open(p1, "rb") as r1: shutil.copyfileobj(r1, out1)
                with open(p2, "rb") as r2: shutil.copyfileobj(r2, out2)
                p1.unlink(); p2.unlink()

        # final FASTQ counts
        fwd_tot = sum(1 for _ in open(fq1_path)) // 4
        rev_tot = sum(1 for _ in open(fq2_path)) // 4
        print(f"Total reads: +={fwd_tot}, -={rev_tot}", file=sys.stderr)
        if fwd_tot != rev_tot:
            raise RuntimeError(f"Final mismatch: {fwd_tot} vs {rev_tot}")
        if fwd_tot != args.read_depth:
            print(f"Warning: generated {fwd_tot} ≠ requested {args.read_depth}", file=sys.stderr)

        # 5) merge depth tables
        depth_df = parts[0]
        for dfc in parts[1:]:
            depth_df = depth_df.add(dfc, fill_value=0).astype(int)
        depth_df.to_csv(depth_path)

        emp_cov = depth_df.values.sum() * args.read_length * 2 / total_bp
        print(f"Coverage for {tag}: {emp_cov:.2f}×", file=sys.stderr)

    print("\nAll simulations complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
