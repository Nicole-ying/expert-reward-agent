import json
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    raise RuntimeError("PyYAML is required to load this config. Please run: pip install pyyaml")


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def make_run_dir(cfg, run_name):
    run_dir = Path(cfg["experiment"]["run_root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["llm_inputs", "human_review", "final_outputs", "raw_outputs", "validations", "prompt_records", "response_records"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def _as_markdown_block(title, content, language=""):
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, indent=2)
        language = "json"
    fence = f"```{language}" if language else "```"
    return f"## {title}\n\n{fence}\n{content}\n```\n"


def record_prompt(run_dir, name, system_prompt, user_prompt):
    """Save prompt records as Markdown instead of JSON so they are readable during debugging."""
    p = Path(run_dir) / "prompt_records" / f"{name}.md"
    text = "# Prompt Record\n\n"
    text += _as_markdown_block("System Prompt", system_prompt, "text")
    text += "\n"
    text += _as_markdown_block("User Prompt", user_prompt, "markdown")
    write_text(p, text)


def record_response(run_dir, name, response):
    """Save response records as Markdown. Structured outputs are embedded as JSON code blocks only when unavoidable."""
    p = Path(run_dir) / "response_records" / f"{name}.md"
    text = "# Response Record\n\n"
    if isinstance(response, str):
        text += response
        if not text.endswith("\n"):
            text += "\n"
    else:
        text += _as_markdown_block("Response", response, "json")
    write_text(p, text)
