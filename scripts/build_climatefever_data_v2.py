import os, re, random, json, csv
from pathlib import Path
from datetime import datetime, timedelta
from datasets import load_dataset
from tqdm import tqdm
import argparse

random.seed(42)

OUT_DOCS = Path("data/docs.csv")
OUT_DIALOGS = Path("data/dialogs.jsonl")

# ---- date helpers ----
MONTHS = ("January","February","March","April","May","June","July","August","September","October","November","December")
ISO_RE   = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
MONTH_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+([1-9]|[12]\d|3[01]),\s*(20\d{2})\b", re.IGNORECASE)
MONTH2NUM = {m.lower(): f"{i:02d}" for i,m in enumerate(MONTHS, start=1)}

def to_iso_dates(text: str):
    out = []
    for y,m,d in ISO_RE.findall(text):
        out.append(f"{y}-{m}-{d}")
    for mon, day, year in MONTH_RE.findall(text):
        out.append(f"{year}-{MONTH2NUM[mon.lower()]}-{str(day).zfill(2)}")
    return out

def shift_days(iso: str, days: int) -> str:
    try:
        dt = datetime.strptime(iso, "%Y-%m-%d") + timedelta(days=days)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

# ---- noise helpers ----
RELIAB_ORDER = ["Low","Medium","High"]
RELIAB = {"wikipedia_revision":"High","dept_site":"High","lms_post":"Medium","student_rumor":"Low","peer_chat":"Low"}

def perturb_reliability(label: str, p: float) -> str:
    """With prob p, move reliability one step up/down (bounded)."""
    if random.random() >= p: 
        return label
    i = RELIAB_ORDER.index(label) if label in RELIAB_ORDER else 1
    step = random.choice([-1, 1])
    j = max(0, min(2, i + step))
    return RELIAB_ORDER[j]

def maybe_drop(value: str, p: float) -> str:
    return "" if (value and random.random() < p) else value

def close_time_to_doc(doc_iso: str, max_delta=2) -> str:
    """Return a date within ±max_delta days of doc date."""
    if not doc_iso:
        return ""
    d = random.randint(-max_delta, max_delta)
    return shift_days(doc_iso, d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=100, help="Total samples to generate")
    ap.add_argument("--ratio", type=str, default="40,40,10,10",
                    help="Counts for MemTrue_RAGStale,RAGTrue_MemRumor,Unknown,Edge (sum must equal total)")
    # noise configs
    ap.add_argument("--p_reliab_doc", type=float, default=0.10, help="Doc reliability perturb prob")
    ap.add_argument("--p_reliab_mem", type=float, default=0.15, help="Mem reliability perturb prob")
    ap.add_argument("--p_drop_time_doc", type=float, default=0.10, help="Drop time_doc prob")
    ap.add_argument("--p_drop_time_mem", type=float, default=0.20, help="Drop time_mem prob")
    ap.add_argument("--p_close_time_edge", type=float, default=0.70, help="Edge: use time close to doc prob")
    ap.add_argument("--p_mem_flip_rumor", type=float, default=0.25, help="Rumor says old value or near-old prob")
    args = ap.parse_args()

    # splits
    r = [int(x) for x in args.ratio.split(",")]
    assert sum(r) == args.total, "ratio must sum to total"
    SPLIT = {
        "MemTrue_RAGStale": r[0],
        "RAGTrue_MemRumor": r[1],
        "Unknown": r[2],
        "Edge": r[3],
    }

    os.makedirs("data", exist_ok=True)
    # Climate-FEVER has a single 'test' split
    ds = load_dataset("tdiggelm/climate_fever", split="test")

    # Keep items with evidence containing dates
    usable = []
    for ex in tqdm(ds, desc="Filter claims with dated evidence"):
        claim = ex.get("claim","")
        evs = ex.get("evidences") or ex.get("evidence") or []
        if not (claim and isinstance(evs, list)):
            continue
        texts = []
        for it in evs:
            if isinstance(it, dict):
                s = it.get("evidence") or it.get("evidence_text") or ""
                if isinstance(s, str) and s.strip():
                    texts.append(s.strip())
            elif isinstance(it, str) and it.strip():
                texts.append(it.strip())
        if not texts:
            continue
        ev_text = " ".join(texts)
        if len(ev_text) < 60:
            continue
        dates = to_iso_dates(ev_text)
        if not dates:
            continue
        value_doc = dates[0]
        time_doc  = value_doc
        title = ex.get("evidence_page","") or ex.get("evidence_wiki_url","") or "Wikipedia"
        usable.append({
            "title": title,
            "claim": claim,
            "value_doc": value_doc,
            "time_doc": time_doc,
            "text_doc": ev_text[:1200]
        })
    need = args.total * 2
    if len(usable) < need:
        print(f"[warn] only {len(usable)} usable; generation may stop early")
    random.shuffle(usable)

    # round-robin fill
    counts = {k:0 for k in SPLIT}
    def pick_bucket():
        for k in ("MemTrue_RAGStale","RAGTrue_MemRumor","Unknown","Edge"):
            if counts[k] < SPLIT[k]:
                return k
        return None

    docs_rows, dialogs_rows = [], []
    used, i = 0, 0
    while used < args.total and i < len(usable):
        ex = usable[i]; i += 1
        bucket = pick_bucket()
        if not bucket: break

        course = f"CLM{used:03d}"
        dialog_id = f"{course.lower()}_{bucket.lower()}"

        # retrieval (doc)
        value_doc = ex["value_doc"]
        time_doc  = ex["time_doc"]
        source_doc = "wikipedia_revision"
        reliab_doc = RELIAB[source_doc]
        # noise: doc reliability & time
        reliab_doc = perturb_reliability(reliab_doc, args.p_reliab_doc)
        time_doc_out = maybe_drop(time_doc, args.p_drop_time_doc)

        title = ex["title"] or "Wikipedia"
        text_doc = f"{title} — Evidence:\n{ex['text_doc']}\n\nRecorded date: {value_doc}."

        # memory update by bucket
        if bucket == "MemTrue_RAGStale":
            source_mem = "wikipedia_revision"; reliab_mem = RELIAB[source_mem]
            value_mem = shift_days(value_doc, random.choice([7,14,21,30]))
            time_mem  = value_mem
            # noise
            reliab_mem = perturb_reliability(reliab_mem, args.p_reliab_mem)
            time_mem_out = maybe_drop(time_mem, args.p_drop_time_mem)
            update_text = f"Update (Wikipedia): the date is now {value_mem}" + (f" ({time_mem_out})." if time_mem_out else ".")
            gold = value_mem if time_mem_out or reliab_mem in ("High","Medium") else value_doc

        elif bucket == "RAGTrue_MemRumor":
            source_mem = "student_rumor"; reliab_mem = RELIAB[source_mem]
            # rumor：多数情况下改成远离doc；小概率“翻车”说回旧值或逼近旧值，制造模糊
            if random.random() < args.p_mem_flip_rumor:
                # 回到 doc 或接近 doc（±1～2 天）
                value_mem = random.choice([value_doc, shift_days(value_doc, random.choice([-2,-1,1,2]))])
            else:
                value_mem = shift_days(value_doc, random.choice([-30,-21,-14,-7,7,14,21,30]))
            time_mem_out = maybe_drop("", args.p_drop_time_mem)  # 传言通常无时间
            update_text = f"I heard someone said the date changed to {value_mem} (not sure)."
            gold = value_doc  # RAG 为准

        elif bucket == "Unknown":
            source_mem = "peer_chat"; reliab_mem = RELIAB[source_mem]
            value_mem = ""
            time_mem_out = ""
            update_text = "A friend mentioned it might be moved next month."
            gold = "unknown"

        else:  # Edge
            # 来源可能是 lms_post (Med) 或 wikipedia_revision (High)
            source_mem = random.choice(["lms_post","wikipedia_revision"])
            reliab_mem = RELIAB[source_mem]
            reliab_mem = perturb_reliability(reliab_mem, args.p_reliab_mem)
            # 时间接近或随机
            if random.random() < args.p_close_time_edge:
                value_mem = close_time_to_doc(value_doc, max_delta=2)
            else:
                value_mem = shift_days(value_doc, random.choice([-5,-4,-3,3,4,5]))
            # 时间可能缺失
            time_mem_out = maybe_drop(value_mem, args.p_drop_time_mem)
            update_text = f"New post: the date is {value_mem}" + (f" ({time_mem_out})." if time_mem_out else ".")
            # gold：若无时间更可能 unknown；否则高可靠优先，其次更近/更晚并不绝对
            if not time_mem_out:
                gold = "unknown"
            else:
                if source_mem == "wikipedia_revision" and value_mem != value_doc:
                    gold = value_mem
                else:
                    # 更保守：保持 doc
                    gold = value_doc

        turns = [
            {"role":"user","text": f"About {title}, when did this happen?"},
            {"role":"assistant","text":"The article includes a recorded date in its evidence."},
            {"role":"user","text": update_text},
            {"role":"user","text":"So what is the date now?"}
        ]

        docs_rows.append({
            "doc_id": f"d{used+1:03d}",
            "course": course,
            "slot": "date",
            "value_doc": value_doc,
            "time_doc": time_doc_out,
            "source_doc": source_doc,
            "reliability_doc": reliab_doc,
            "title": title,
            "text": text_doc
        })

        dialogs_rows.append({
            "dialog_id": dialog_id,
            "course": course,
            "turns": turns,
            "update": {
                "slot": "date",
                "value_mem": value_mem,
                "time_mem": time_mem_out,
                "source_mem": source_mem,
                "reliability_mem": reliab_mem
            },
            "question": "What is the date now?",
            "gold_truth": gold,
            "evaluation_tag": bucket
        })

        counts[bucket] += 1
        used += 1

    OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIALOGS.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_DOCS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "doc_id","course","slot","value_doc","time_doc",
            "source_doc","reliability_doc","title","text"
        ])
        w.writeheader()
        for r in docs_rows:
            w.writerow(r)

    with open(OUT_DIALOGS, "w", encoding="utf-8") as f:
        for r in dialogs_rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

    print("Done.")
    print("Counts:", counts)
    print(f"Wrote {OUT_DOCS} and {OUT_DIALOGS}")
    print("Noise config:",
          {"p_reliab_doc": args.p_reliab_doc,
           "p_reliab_mem": args.p_reliab_mem,
           "p_drop_time_doc": args.p_drop_time_doc,
           "p_drop_time_mem": args.p_drop_time_mem,
           "p_close_time_edge": args.p_close_time_edge,
           "p_mem_flip_rumor": args.p_mem_flip_rumor})

if __name__ == "__main__":
    main()
