import argparse, json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from src.decider import always_rag, always_mem, rule_weighted, conservative

POLICIES = {
    "rag_only": always_rag,
    "mem_only": always_mem,
    "rule": lambda d,u: rule_weighted(d,u,delta_days=5),
    "cons": lambda d,u: conservative(d,u,delta_days=5),
}

def load_docs(path_csv):
    df = pd.read_csv(path_csv)
    required = {
        "doc_id","course","slot","value_doc","time_doc",
        "source_doc","reliability_doc","title","text"
    }
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"docs.csv missing: {missing}")
    # map course → first row (one doc per course)
    docs = {r["course"]: r._asdict() if hasattr(r, "_asdict") else r.to_dict()
            for _, r in df.iterrows()}
    return docs

def load_dialogs(path_jsonl):
    items = []
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            obj = json.loads(line)
            for k in ["dialog_id","course","turns","question","gold_truth","evaluation_tag"]:
                if k not in obj:
                    raise ValueError(f"dialogs.jsonl sample missing {k}")
            items.append(obj)
    return items

def compute_metrics(rows):
    df = pd.DataFrame(rows)
    df["em"] = (df["answer"].astype(str).str.strip()
                == df["gold_truth"].astype(str).str.strip()).astype(int)
    df["abstain"] = (df["answer"].astype(str).str.lower()=="unknown").astype(int)

    overall = (df.groupby("policy")
                 .agg(n=("em","count"),
                      em_mean=("em","mean"),
                      abstain_rate=("abstain","mean"))
                 .reset_index())

    by_tag = (df.groupby(["policy","evaluation_tag"])
                .agg(n=("em","count"),
                     em_mean=("em","mean"),
                     abstain_rate=("abstain","mean"))
                .reset_index())

    return df, overall, by_tag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs.csv")
    ap.add_argument("--dialogs", default="data/dialogs.jsonl")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    docs = load_docs(args.docs)
    dlgs = load_dialogs(args.dialogs)

    outdir = Path(args.outdir or Path("results/runs"))
    outdir.mkdir(parents=True, exist_ok=True)
    run_dir = outdir / pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in tqdm(dlgs, desc="Eval"):
        course = item["course"]
        doc = docs.get(course)
        if doc is None:
            continue
        for name, fn in POLICIES.items():
            ans = fn(doc, item.get("update", {}))
            rows.append({
                "dialog_id": item["dialog_id"],
                "course": course,
                "policy": name,
                "gold_truth": item["gold_truth"],
                "evaluation_tag": item["evaluation_tag"],
                "answer": ans
            })

    df, overall, by_tag = compute_metrics(rows)
    df.to_csv(run_dir/"predictions.csv", index=False)
    overall.to_json(run_dir/"metrics_overall.json", orient="records", indent=2)
    by_tag.to_json(run_dir/"metrics_by_tag.json", orient="records", indent=2)

    print(f"[OK] wrote {run_dir/'predictions.csv'}")
    print("\nOverall:")
    print(overall.to_string(index=False))
    print("\nBy scenario (evaluation_tag):")
    print(by_tag.to_string(index=False))

if __name__ == "__main__":
    main()
