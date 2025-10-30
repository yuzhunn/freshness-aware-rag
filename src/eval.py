import argparse, os, json, re, csv, time, hashlib, datetime as dt
from pathlib import Path
import pandas as pd
from collections import defaultdict

# DATE_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
# two patterns of date
ISO_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
MONTH_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"([1-9]|[12]\d|3[01]),\s*(20\d{2})\b",
    re.IGNORECASE
)
MONTH2NUM = {
    "january":"01","february":"02","march":"03","april":"04","may":"05","june":"06",
    "july":"07","august":"08","september":"09","october":"10","november":"11","december":"12"
}

def extract_all_dates_iso(text: str):
    """Return a list of dates in ISO YYYY-MM-DD from both ISO and 'Month DD, YYYY' forms, in order of appearance."""
    out = []
    # ISO first, keep order
    for y,m,d in ISO_RE.findall(text):
        out.append(f"{y}-{m}-{d}")
    # Month DD, YYYY -> ISO
    for mon, day, year in MONTH_RE.findall(text):
        mm = MONTH2NUM[mon.lower()]
        dd = str(day).zfill(2)
        out.append(f"{year}-{mm}-{dd}")
    return out


def nowstamp():
    return dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")

def load_docs(docs_csv):
    df = pd.read_csv(docs_csv)
    # sanity checks
    required = {"doc_id","course","slot","old_value","date_old","title","text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"docs.csv missing columns: {missing}")
    # one row per course (your data fits this); if multiple, we pick the first
    docs = {}
    for _, r in df.iterrows():
        docs.setdefault(r["course"], []).append(dict(r))
    return docs

def load_dialogs(dialogs_jsonl):
    items = []
    with open(dialogs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            items.append(json.loads(line))
    # sanity
    for it in items:
        for k in ["dialog_id","course","turns","question","gold_latest_value"]:
            if k not in it:
                raise ValueError(f"dialogs.jsonl item missing key: {k} // {it.get('dialog_id')}")
    return items

# --------- Context composers ---------

def build_rag_context(doc_row, tail_turns):
    doc_block = f"Title: {doc_row['title']}\nText: {doc_row['text']}\n"
    turns_block = "\n".join([f"{t['role'].capitalize()}: {t['text']}" for t in tail_turns])
    return doc_block + "\n" + turns_block

def extract_latest_update_from_dialog(turns):
    latest = None
    for t in turns:
        dates = extract_all_dates_iso(t["text"])
        if dates:
            latest = dates[-1]  # keep the most recent mention in reading order
    return latest

def build_memory_line(course, latest_value):
    if latest_value:
        return f"Memory: {course}.deadline = {latest_value} (updated)"
    else:
        return "Memory: UNRESOLVED"

def build_lw_context(course, doc_row, turns):
    latest = extract_latest_update_from_dialog(turns)
    memory_line = build_memory_line(course, latest)
    doc_block = f"Title: {doc_row['title']}\nText: {doc_row['text']}\n"
    turns_block = "\n".join([f"{t['role'].capitalize()}: {t['text']}" for t in turns[-2:]])  # last 1–2 turns
    return memory_line + "\n\n" + doc_block + "\n" + turns_block, latest

# --------- "Answer engines" (no-LLM default) ----------

def regex_answer_from_context(context, prefer_first_doc_date=True):
    # Try to isolate the doc text block
    text_section = None
    if "Text:" in context:
        text_section = context.split("Text:", 1)[1]

    if prefer_first_doc_date and text_section:
        doc_dates = extract_all_dates_iso(text_section)
        if doc_dates:
            return doc_dates[0]  # mimic RAG trusting the doc (old value)

    # fallback: scan whole context
    all_dates = extract_all_dates_iso(context)
    if not all_dates:
        return "unknown"
    return all_dates[0] if prefer_first_doc_date else all_dates[-1]


def answer_rag_only(doc_row, turns, question):
    # RAG-only context: doc + last 1–2 turns; answer picks first doc date if present (tends to old value)
    context = build_rag_context(doc_row, turns[-2:])
    ans = regex_answer_from_context(context, prefer_first_doc_date=True)
    return ans, context

def answer_latest_wins(course, doc_row, turns, question):
    # LW context: memory on top; if memory resolved, answer = that date; else fallback like rag-only
    context, latest = build_lw_context(course, doc_row, turns)
    if latest:
        ans = latest
    else:
        ans = regex_answer_from_context(context, prefer_first_doc_date=False)
    return ans, context

# --------- Metrics ----------

def compute_metrics(pred_rows):
    """
    pred_rows: list of dict with keys:
      dialog_id, course, policy, old_value, gold_latest, answer
    Adds em (0/1) and stale (0/1).
    """
    for r in pred_rows:
        r["em"] = int(str(r["answer"]).strip() == str(r["gold_latest"]).strip())
        r["stale"] = int(str(r["answer"]).strip() == str(r["old_value"]).strip())
    df = pd.DataFrame(pred_rows)
    summ = {}
    for pol in df["policy"].unique():
        sub = df[df["policy"]==pol]
        summ[pol] = {
            "n": int(len(sub)),
            "em_mean": float(sub["em"].mean()) if len(sub) else 0.0,
            "stale_mean": float(sub["stale"].mean()) if len(sub) else 0.0,
        }
    overall = {
        "n_total": int(len(df)),
        "policies": summ,
    }
    return df, overall

# --------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs.csv")
    ap.add_argument("--dialogs", default="data/dialogs.jsonl")
    ap.add_argument("--policy", choices=["rag_only","latest_wins","both"], default="both")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    docs = load_docs(args.docs)
    dialogs = load_dialogs(args.dialogs)

    stamp = nowstamp()
    outdir = Path(args.outdir or f"results/runs/{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    pred_rows = []

    for item in dialogs:
        course = item["course"]
        dialog_id = item["dialog_id"]
        question = item["question"]
        gold = item["gold_latest_value"]

        # pick first doc for this course
        doc_list = docs.get(course, [])
        if not doc_list:
            # skip if missing (should not happen with your data)
            continue
        doc_row = doc_list[0]
        old_value = doc_row["old_value"]

        # which policies to run
        policies = ["rag_only","latest_wins"] if args.policy=="both" else [args.policy]

        for pol in policies:
            if pol == "rag_only":
                ans, ctx = answer_rag_only(doc_row, item["turns"], question)
            else:
                ans, ctx = answer_latest_wins(course, doc_row, item["turns"], question)

            pred_rows.append({
                "dialog_id": dialog_id,
                "course": course,
                "policy": pol,
                "old_value": old_value,
                "gold_latest": gold,
                "answer": ans,
                "context_hash": hashlib.md5(ctx.encode("utf-8")).hexdigest()[:8],
            })

    # compute metrics
    df_preds, metrics = compute_metrics(pred_rows)

    # save
    preds_path = outdir / "predictions.csv"
    metrics_path = outdir / "metrics.json"
    df_preds.to_csv(preds_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] wrote {preds_path}")
    print(f"[OK] wrote {metrics_path}")
    print("\nSummary:")
    for pol, m in metrics["policies"].items():
        print(f"  {pol:12s}  n={m['n']:2d}  EM={m['em_mean']:.3f}  stale={m['stale_mean']:.3f}")

if __name__ == "__main__":
    main()
