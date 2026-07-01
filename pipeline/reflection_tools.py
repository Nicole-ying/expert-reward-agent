"""Retrieval tools for the reflection agent — simple keyword-based, no ChromaDB needed."""

import json
import re
from pathlib import Path

_techniques_text = None
_skeletons_data = None


def _load_techniques():
    global _techniques_text
    if _techniques_text is None:
        p = Path("knowledge_base/iteration/reward_design_techniques.md")
        _techniques_text = p.read_text(encoding="utf-8") if p.exists() else ""
    return _techniques_text


def _load_skeletons():
    global _skeletons_data
    if _skeletons_data is None:
        p = Path("knowledge_base/iteration/skeleton_details.json")
        _skeletons_data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _skeletons_data


def _split_sections(md_text):
    """Split markdown into per-##-heading sections."""
    sections = []
    current = []
    for line in md_text.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return sections


def search_reward_design_knowledge(query: str) -> str:
    """Search the technique library for matching entries. Returns top 2 matches."""
    text = _load_techniques()
    sections = _split_sections(text)
    if not sections:
        return "(技法库为空)"
    query_lower = query.lower()
    scored = []
    for sec in sections:
        sec_lower = sec.lower()
        # Score: keyword match count in section
        keywords = re.findall(r"[a-z_]{3,}", query_lower)
        score = sum(1 for kw in keywords if kw in sec_lower)
        # Bonus for heading match
        heading = sec.split("\n")[0].lower() if sec else ""
        score += sum(2 for kw in keywords if kw in heading)
        if score > 0:
            scored.append((score, sec))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return "(未找到匹配的技法条目)"
    # Return top 2, keep it compact
    results = []
    for _, sec in scored[:2]:
        # Extract just the heading + 症状 + 修复 lines
        lines = sec.splitlines()
        compact = []
        for line in lines:
            s = line.strip()
            if s.startswith("## ") or s.startswith("- 症状") or s.startswith("- 修复") or s.startswith("- 原理"):
                compact.append(s)
        results.append("\n".join(compact) if compact else lines[0])
    return "\n---\n".join(results)


def get_skeleton_detail(skeleton_name: str) -> str:
    """Get the math form, rationale, pitfalls, and usage of a skeleton."""
    skeletons = _load_skeletons()
    sk = skeletons.get(skeleton_name)
    if not sk:
        # Try fuzzy match
        for key in skeletons:
            if skeleton_name.lower() in key.lower():
                sk = skeletons[key]
                skeleton_name = key
                break
    if not sk:
        return f"骨架 '{skeleton_name}' 未在 skeleton_details.json 中找到。"
    return (
        f"## {skeleton_name}\n"
        f"- 数学形态: {sk['math_form']}\n"
        f"- 设计原理: {sk['design_rationale']}\n"
        f"- 常见陷阱: {sk['pitfalls']}\n"
        f"- 推荐配合: {sk['recommended_with']}"
    )
