# Freshness-Aware Decision Mechanisms in Retrieval-Augmented Memory Agents

## Overview
This project investigates how Large Language Model (LLM) agents can make better decisions when their long-term memory and retrieved documents contain conflicting information.  
The focus is on evaluating simple, interpretable **freshness-aware decision mechanisms** that consider both the *recency* and *reliability* of information sources.

---

## Research Question
When retrieval and memory conflict, how should an LLM agent decide which to trust — the older, reliable source or the newer, potentially noisy update?

---

## Method

### Dataset
- Based on the **Climate-FEVER** dataset (40 examples).  
- Each example contains:
  - A retrieved fact
  - A memory update (sometimes conflicting or unreliable)
  - Ground truth and reliability labels

### Scenarios
- `MemTrue_RAGStale`: memory correct, retrieval outdated  
- `RAGTrue_MemRumor`: retrieval correct, memory incorrect  
- `Unknown`: incomplete or ambiguous  
- `Edge`: borderline freshness conflict

### Decision Policies
| Policy | Description |
|---------|-------------|
| `rag_only` | Always trust retrieval |
| `mem_only` | Always trust memory |
| `rule` | Weighted by reliability and timestamp |
| `cons` | Same as rule, but abstains when uncertain |

---

## Running the Experiment

### Setup
```bash
git clone https://github.com/yourname/freshness-aware-rag.git
cd freshness-aware-rag
pip install -r requirements.txt
```

```Run Evaluation
python -m src.eval --docs data/docs.csv --dialogs data/dialogs.jsonl --policy both
```

```Generate Figures
python scripts/make_figures.py --runs results/runs/* --out results/figures/
```

## Conclusion

Freshness-aware decision rules can improve factual reliability in retrieval-augmented memory agents.
Explicit reasoning about when to trust memory vs. retrieval helps reduce hallucinations and outdated responses.

Future work will explore learnable decision policies and larger, real-world datasets.