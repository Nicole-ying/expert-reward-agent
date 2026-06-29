import json
from pathlib import Path


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def markdown_block(title, obj_or_text):
    if isinstance(obj_or_text, str):
        body = obj_or_text
    else:
        body = "```json\n" + json.dumps(obj_or_text, ensure_ascii=False, indent=2) + "\n```"
    return f"# {title}\n\n{body}\n"
