"""Subagent Investigator - training dynamics evidence scout.

Reads training process data (monitor logs, component stats, final evaluation)
and produces a compact evidence report. Does NOT make decisions or suggest edits.
The reflection LLM owns all decisions.

Key design: this is a TRAINING PROCESS observer, not just a final-eval reader.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List


# -- Data readers ----------------------------------------------------------

def _read_monitor_trends(train_dir: str) -> str:
    """Read monitor CSV files and extract training progress trends."""
    monitor_dir = Path(train_dir) / "monitor"
    if not monitor_dir.exists():
        return "(无monitor数据)"

    all_episodes = []
    for csv_file in sorted(monitor_dir.glob("*.csv")):
        try:
            lines = csv_file.read_text(encoding="utf-8").splitlines()
            for line in lines[1:]:  # skip header
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    try:
                        r = float(parts[0])
                        l = float(parts[1])
                        orig = float(parts[3]) if len(parts) > 3 else 0.0
                        gen = float(parts[4]) if len(parts) > 4 else 0.0
                        all_episodes.append({"r": r, "l": l, "orig": orig, "gen": gen})
                    except (ValueError, IndexError):
                        continue
        except Exception:
            continue

    if len(all_episodes) < 5:
        return "(monitor数据不足)"

    # Split into thirds: early, mid, late training
    n = len(all_episodes)
    third = n // 3
    early = all_episodes[:max(1, third)]
    mid = all_episodes[third:2*third]
    late = all_episodes[2*third:]

    def summarize(eps, label):
        lens = [e["l"] for e in eps]
        origs = [e["orig"] for e in eps]
        gens = [e["gen"] for e in eps]
        rs = [e["r"] for e in eps]
        if not lens:
            return ""
        avg_len = sum(lens) / len(lens)
        avg_orig = sum(origs) / len(origs)
        avg_gen = sum(gens) / len(gens)
        avg_r = sum(rs) / len(rs)
        crash_rate = sum(1 for l in lens if l < 150) / len(lens)
        return (f"{label}({len(eps)}局): "
                f"avg_len={avg_len:.0f}, avg_score={avg_r:.1f}, "
                f"gen_reward={avg_gen:.3f}/step, orig_reward={avg_orig:.3f}/step, "
                f"crash_rate={crash_rate:.0%}")

    lines = ["## 训练过程趋势（monitor日志）", ""]
    lines.append(summarize(early, "早期"))
    lines.append(summarize(mid, "中期"))
    lines.append(summarize(late, "后期"))

    # Detect patterns
    early_len = sum(e["l"] for e in early) / max(1, len(early))
    late_len = sum(e["l"] for e in late) / max(1, len(late))
    early_crash = sum(1 for e in early if e["l"] < 150) / max(1, len(early))
    late_crash = sum(1 for e in late if e["l"] < 150) / max(1, len(late))

    patterns = []
    if late_len > early_len * 3:
        patterns.append("episode长度显著增长: agent从早期快速失败过渡到后期长时间存活")
    if early_crash > 0.8 and late_crash < 0.3:
        patterns.append("agent学会了避免早期crash: crash率从{:.0%}降至{:.0%}".format(early_crash, late_crash))
    if late_crash > 0.7:
        patterns.append("agent始终未能学会生存: 后期crash率仍>{:.0%}".format(late_crash))
    if late_len > 900 and late_crash < 0.2:
        patterns.append("agent学会了长时间存活但可能未完成任务(悬停): ep_len>900且crash率低")
    if patterns:
        lines.append("")
        lines.append("**关键趋势**:")
        for p in patterns:
            lines.append(f"- {p}")

    return "\n".join(lines)


def _read_training_summary(train_dir: str) -> str:
    """Read training_summary.json: eval results + training component stats."""
    p = Path(train_dir) / "training_summary.json"
    if not p.exists():
        return "(training_summary.json not found)"

    data = json.loads(p.read_text(encoding="utf-8"))
    external = data.get("external_eval", {})
    comp_summary = data.get("component_summary", {})
    comp_stats = comp_summary.get("component_stats", {})

    lines = ["## 最终评估结果",
             f"mean_eval_reward={external.get('mean_eval_reward','?')}",
             f"mean_episode_length={external.get('mean_episode_length','?')}",
             f"terminated={sum(1 for t in external.get('episode_terminated',[]) if t)}/{len(external.get('episode_terminated',[]) or [])}",
             f"reward_range=[{external.get('min_eval_reward','?')}, {external.get('max_eval_reward','?')}]",
             "",
             "## 训练期组件统计（全训练过程聚合）"]

    # Sort components by contribution
    sorted_comps = sorted(comp_stats.items(),
                          key=lambda x: abs(x[1].get('mean', 0)) * x[1].get('nonzero_rate', 0),
                          reverse=True)

    for name, stats in sorted_comps:
        short = name.replace("component.", "")
        mean_val = stats.get('mean', 0)
        nonzero = stats.get('nonzero_rate', 0)
        mean_active = stats.get('mean_when_active', 0)
        lines.append(f"  {short}: per_step_mean={mean_val:.4f}, "
                     f"nonzero_rate={nonzero:.1%}, "
                     f"mean_when_active={mean_active:.4f}")

    # Compare training vs eval
    lines.append("")
    lines.append("## 训练 vs 评估 对比")
    lines.append("(训练期组件数据来自PPO训练步, 评估数据来自最终20局确定性策略)")
    train_gen = comp_stats.get("component.generated_reward", {})
    eval_orig = external.get('mean_eval_reward', 0)
    train_orig = comp_stats.get("component.original_env_reward", {})

    if train_gen and train_orig:
        train_gen_mean = train_gen.get('mean', 0)
        train_orig_mean = train_orig.get('mean', 0)
        lines.append(f"  训练期generated_reward均值: {train_gen_mean:.4f}/step")
        lines.append(f"  训练期original_reward均值: {train_orig_mean:.4f}/step")
        lines.append(f"  最终评估original_reward: {eval_orig:.1f}")

    # Dead component check
    dead = [(n, s) for n, s in comp_stats.items()
            if s.get('nonzero_rate', 0) < 0.02 and 'generated' not in n and 'original' not in n]
    if dead:
        lines.append("")
        lines.append("**训练期几乎不触发的组件**:")
        for name, stats in dead:
            short = name.replace("component.", "")
            lines.append(f"  {short}: nonzero_rate={stats['nonzero_rate']:.1%} (可能是僵尸组件)")

    return "\n".join(lines)


def _read_eval_detail(train_dir: str) -> str:
    """Read per-episode evaluation breakdown."""
    p = Path(train_dir) / "eval_result.md"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    # Extract just the per-episode table
    in_table = False
    episodes = []
    for line in lines:
        if line.startswith("| episode"):
            in_table = True
            continue
        if in_table and line.startswith("|") and "---" not in line:
            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) >= 5:
                episodes.append(f"  ep{cols[0]}: score={cols[2]}, len={cols[3]}, {cols[4]}")
        elif in_table and not line.startswith("|"):
            break

    if not episodes:
        return ""

    # Show worst and best episodes
    return "## 评估局详情（最差/最好各3局）\n" + "\n".join(episodes[:3] + ["  ..."] + episodes[-3:])


def _read_previous_reward(reward_path: str) -> str:
    if not reward_path or not Path(reward_path).exists():
        return "(not available)"
    return Path(reward_path).read_text(encoding="utf-8")[:4000]


# -- Main entry point ------------------------------------------------------

def run_investigator(
    *,
    train_dir: str,
    previous_reward_path: str = "",
    memory_path: str = "",
    client: Any = None,
    model: str = "deepseek-chat",
    max_tokens: int = 2000,
) -> Dict[str, Any]:
    """Single-call evidence scout. Reads training process data and produces
    a diagnostic evidence report for the reflection LLM."""

    # Pre-load all data
    monitor_trends = _read_monitor_trends(train_dir)
    training_summary = _read_training_summary(train_dir)
    eval_detail = _read_eval_detail(train_dir)
    prev_code = _read_previous_reward(previous_reward_path)

    system_prompt = (
        "You are a TRAINING PROCESS OBSERVER. Your job is to read training "
        "data and report WHAT HAPPENED during training — not what should "
        "happen next.\n\n"
        "YOU ARE NOT A DECISION-MAKER. You do not suggest reward edits, "
        "coefficient changes, or operator choices. The reflection LLM "
        "(which sees environment facts, formula operator library, memory, "
        "and best reward code) makes all decisions.\n\n"
        "YOUR VALUE: You see training process data (monitor logs, component "
        "evolution) that the reflection LLM does NOT see. The reflection LLM "
        "only sees final evaluation data. You bridge this gap.\n\n"
        "WHAT TO REPORT:\n"
        "- Training progress: did the agent improve over time? Early vs late patterns?\n"
        "- Reward-reality gap: is generated_reward aligned with original_env_reward?\n"
        "- Component health: which components fired during training? Which were dead?\n"
        "- Training anomalies: early plateau, sudden crash, reward exploitation\n\n"
        "Output valid JSON with these fields:\n"
        "- training_progress: how agent behavior evolved (early→mid→late). "
        "  Report episode length trends, crash rate changes, score trajectory.\n"
        "- component_health: which components were active/inactive during training. "
        "  Report nonzero rates, dominant components, dead components.\n"
        "- reward_alignment: is the shaped reward aligned with task success? "
        "  Compare generated_reward vs original_env_reward. Report exploitation signs.\n"
        "- anomalies: training irregularities (sudden divergence, value explosion, "
        "  early convergence signals). Empty string if none detected.\n"
        "- confidence: low/medium/high (completeness of evidence picture)\n\n"
        "RULES:\n"
        "- Every claim must cite a metric from the data.\n"
        "- Do NOT propose what to change. Do NOT mention coefficients or operators.\n"
        "- Be terse. Total JSON ~500-700 chars.\n"
        "- If data is sparse, say so explicitly and set confidence=low."
    )

    user_prompt = f"""Training data for the most recently completed reward iteration:

=== 训练过程趋势（Monitor日志） ===
{monitor_trends[:4000]}

=== 训练期组件统计 ===
{training_summary[:4000]}

=== 评估细节 ===
{eval_detail[:1000]}

=== 当前奖励函数代码 ===
```python
{prev_code[:2500]}
```

Based on the evidence above, output your JSON evidence report. Facts only, no recommendations."""

    signal = None
    tool_trace = []

    try:
        resp = client.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        content = (resp.choices[0].message.content or "").strip()

        try:
            signal = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                try:
                    signal = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        if signal and isinstance(signal, dict):
            required = ["training_progress", "component_health", "reward_alignment", "anomalies"]
            missing = [f for f in required if not signal.get(f)]
            if missing:
                for f in missing:
                    signal[f] = "Not reported."
            if signal.get("confidence") not in ("low", "medium", "high"):
                signal["confidence"] = "low"
        else:
            signal = None

    except Exception as exc:
        tool_trace.append({"error": f"{type(exc).__name__}: {exc}"})

    # Build signal text
    signal_text = ""
    if signal:
        parts = [
            f"**训练过程**: {signal.get('training_progress', '')}",
            f"**组件健康**: {signal.get('component_health', '')}",
            f"**奖励对齐**: {signal.get('reward_alignment', '')}",
        ]
        if signal.get('anomalies'):
            parts.append(f"**异常检测**: {signal.get('anomalies', '')}")
        parts.append(f"**置信度**: `{signal.get('confidence', 'low')}`")
        signal_text = "\n\n".join(parts)

    return {
        "research_signal": signal,
        "research_signal_text": signal_text,
        "turns_used": 1 if signal else 0,
        "tool_trace": tool_trace,
    }
