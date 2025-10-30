import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_runs(runs_glob):
    # you can pass a directory with one run, or a glob like results/runs/*
    p = Path(runs_glob)
    if p.is_dir():
        yield p
    else:
        for run in Path().glob(runs_glob):
            if run.is_dir():
                yield run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs/*")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # take the latest run only (simplest)
    runs = sorted(load_runs(args.runs))
    if not runs:
        print("No runs found.")
        return
    run = runs[-1]
    preds_path = run / "predictions.csv"
    if not preds_path.exists():
        print(f"No predictions.csv in {run}")
        return

    df = pd.read_csv(preds_path)
    # Aggregate
    agg = df.groupby("policy").agg(em_mean=("em","mean"), stale_mean=("stale","mean"), n=("em","count")).reset_index()
    print(agg)

    # Fig 1: EM
    plt.figure()
    plt.bar(agg["policy"], agg["em_mean"])
    plt.ylim(0,1)
    plt.ylabel("Exact Match")
    plt.title("EM by Policy")
    plt.savefig(outdir / "em_bar.png", bbox_inches="tight", dpi=160)

    # Fig 2: Stale rate
    plt.figure()
    plt.bar(agg["policy"], agg["stale_mean"])
    plt.ylim(0,1)
    plt.ylabel("Stale (predict old value)")
    plt.title("Stale Rate by Policy")
    plt.savefig(outdir / "stale_bar.png", bbox_inches="tight", dpi=160)

    print(f"[OK] figures saved to {outdir}")

if __name__ == "__main__":
    main()
