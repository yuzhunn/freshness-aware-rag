# dialogs.jsonl schema
- dialog_id: unique id
- course: string
- turns: list of {role: "user"|"assistant", text: str}; include one mid-turn update
- question: final short question
- gold_latest_value: YYYY-MM-DD (the updated truth)
