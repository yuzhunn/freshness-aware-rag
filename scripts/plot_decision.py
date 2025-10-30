import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=False, default=None,
                    help="Path to a specific run dir (results/runs/2025-..). If omitted, uses latest.")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    runs_root = Path("results/runs")
    run_dir = Path(args.run) if args.run else sorted(runs_root.glob("*"))[-1]
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(run_dir/"predictions.csv")
    preds["em"] = (preds["answer"].astype(str).str.strip()
                   == preds["gold_truth"].astype(str).str.strip()).astype(int)
    preds["abstain"] = (preds["answer"].astype(str).str.lower()=="unknown").astype(int)

    overall = preds.groupby("policy").agg(em_mean=("em","mean"),
                                          abstain_rate=("abstain","mean"),
                                          n=("em","count")).reset_index()

    # EM overall
    plt.figure()
    plt.bar(overall["policy"], overall["em_mean"])
    plt.ylim(0,1); plt.ylabel("Exact Match"); plt.title("Overall EM by Policy")
    plt.savefig(outdir/"em_overall.png", dpi=160, bbox_inches="tight")

    # EM by scenario
    by_tag = preds.groupby(["policy","evaluation_tag"]).agg(em_mean=("em","mean")).reset_index()
    tags = sorted(by_tag["evaluation_tag"].unique())
    policies = overall["policy"].tolist()
    width = 0.18
    xs = range(len(tags))
    plt.figure()
    for i, p in enumerate(policies):
        ys = [by_tag[(by_tag.policy==p)&(by_tag.evaluation_tag==t)]["em_mean"].values[0]
              if not by_tag[(by_tag.policy==p)&(by_tag.evaluation_tag==t)].empty else 0.0
              for t in tags]
        plt.bar([x + i*width for x in xs], ys, width=width, label=p)
    plt.ylim(0,1); plt.ylabel("EM"); plt.title("EM by Scenario")
    plt.xticks([x + (len(policies)-1)*width/2 for x in xs], tags, rotation=20)
    plt.legend()
    plt.savefig(outdir/"em_by_tag.png", dpi=160, bbox_inches="tight")

    # Correct abstention (only where gold=unknown)
    unk = preds[preds["gold_truth"].astype(str).str.lower()=="unknown"]
    if len(unk):
        ca = (unk.groupby("policy")
                 .agg(correct_abstain=("abstain","mean"), n=("abstain","count"))
                 .reset_index())
        plt.figure()
        plt.bar(ca["policy"], ca["correct_abstain"])
        plt.ylim(0,1); plt.ylabel("Correct Abstention (gold=unknown)"); plt.title("Abstention Quality")
        plt.savefig(outdir/"abstain_quality.png", dpi=160, bbox_inches="tight")

    print(f"[OK] figures saved to {outdir} (from run {run_dir})")

if __name__ == "__main__":
    main()
