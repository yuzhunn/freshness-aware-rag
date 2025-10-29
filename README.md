# Freshness-Aware RAG (Course Deadlines)

**Goal:** Test whether a top-pinned "Latest-Wins" memory note from dialogue updates reduces stale answers vs RAG-only when retrieved docs contain outdated facts.

## Structure
- `data/docs.csv`: outdated syllabus documents (DocStore)
- `data/dialogs.jsonl`: short dialogs with a mid-turn update + final question
- `prompts/memory_extraction.txt`: extract latest update as a Memory note
- `prompts/answer.txt`: unified QA prompt for all policies

## Quickstart (coming in Phase 2)
- Run evaluation for RAG-only and Latest-Wins, then plot EM and stale rates.
