"""Single-agent reflection: replaces run_04 (analysis) + run_05 (revision).

The agent reads feedback, memory, environment-specific task/profile context,
and optional tool results. It produces revised reward code.
"""

import argparse
import json
import re
from pathlib import Path

from .common import load_config, read_text, write_text, write_json, record_prompt, record_response
from .reflection_tools import (
    get_reward_transformation,
    get_skeleton_detail,
    search_reward_design_knowledge,
)
from .run_03_direct_reward_generator import extract_code, validate_code
from .subagent_investigator import run_investigator
from llm_clients import create_client


def _parse_reflection_fields(resp_text):
    """Extract structured fields from a reflection agent response.

    Returns dict with keys: selected_level, selected_intervention, falsifiable_hypothesis,
    expected_next_round, main_risk, evidence, behavior_diagnosis.

    Handles both standard numbered-field format and JSON-block format.
    """
    fields = {}

    # ── Try JSON format first (some responses wrap fields in a json block) ──
    json_m = re.search(r'```json\s*\n(.*?)\n```', resp_text, re.DOTALL)
    if json_m:
        try:
            json_fields = json.loads(json_m.group(1))
            for key in ['selected_level', 'selected_intervention', 'falsifiable_hypothesis',
                        'expected_next_round', 'main_risk', 'evidence', 'behavior_diagnosis']:
                if key in json_fields:
                    val = str(json_fields[key])
                    if len(val) > 300:
                        val = val[:297] + "..."
                    fields[key] = val
            if fields:
                return fields
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Standard numbered-field format ──
    for key in ['selected_level', 'selected_intervention', 'falsifiable_hypothesis',
                'expected_next_round', 'main_risk', 'evidence', 'behavior_diagnosis']:
        # Match patterns like: "5. `selected_intervention`：..."  or  "5. selected_intervention: ..."
        m = re.search(
            rf'\d+\.\s*`?{key}`?\s*[：:]\s*(.+?)(?=\n\d+\.\s*`|\n\d+\.\s*\w|\n```|\n\Z)',
            resp_text, re.DOTALL
        )
        if m:
            val = m.group(1).strip()
            # Clean up trailing newlines and markdown artifacts
            val = val.rstrip()
            if len(val) > 300:
                val = val[:297] + "..."
            fields[key] = val
    return fields


def _build_cumulative_record(run_root, prefix, seed, current_iteration, memory_md=""):
    """Build the reflection agent's decision autobiography from all past iterations.

    Parses each past reflection response to extract what was changed, why, and the result.
    Presents a per-iteration narrative that lets the agent see its own patterns —
    oscillations, repeated failures, abandoned breakthroughs — without hard-coded rules.

    Returns empty string if no history (iteration 2 or has no response records).
    """
    if current_iteration <= 2:
        return ""

    base = Path(run_root) / prefix / f"seed_{seed}"

    # ── Collect scores from training_summary.json ──
    scores = {}   # iter -> score
    lengths = {}  # iter -> mean episode length
    for i in range(1, current_iteration):
        ts = base / f"iter_{i:02d}" / "training" / "training_summary.json"
        if ts.exists():
            try:
                d = json.loads(ts.read_text(encoding="utf-8"))
                ee = d.get("external_eval", d)
                rewards = ee.get("episode_rewards", [])
                if rewards:
                    scores[i] = sum(rewards) / len(rewards)
                ep_lens = ee.get("episode_lengths", [])
                if ep_lens:
                    lengths[i] = sum(ep_lens) / len(ep_lens)
            except (json.JSONDecodeError, KeyError):
                continue

    # Also get scores from eval_result.json as fallback
    for i in range(1, current_iteration):
        if i in scores:
            continue
        er = base / f"iter_{i:02d}" / "training" / "eval_result.json"
        if er.exists():
            try:
                d = json.loads(er.read_text(encoding="utf-8"))
                if "mean_eval_reward" in d:
                    scores[i] = d["mean_eval_reward"]
                if "mean_episode_length" in d:
                    lengths[i] = d["mean_episode_length"]
            except (json.JSONDecodeError, KeyError):
                continue

    # ── Build per-iteration narrative ──
    lines = [
        "# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何",
        "",
        "逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。",
        "",
    ]

    iterations_with_data = 0
    component_mod_count = {}  # component -> list of (iter, score_delta)

    for i in range(2, current_iteration):
        resp_path = base / f"iter_{i:02d}" / "generation" / "response_records" / "agent_reflection.md"
        if not resp_path.exists():
            continue

        resp_text = resp_path.read_text(encoding="utf-8")
        fields = _parse_reflection_fields(resp_text)

        intervention = fields.get("selected_intervention", "")
        hypothesis = fields.get("falsifiable_hypothesis", "")
        selected_level = fields.get("selected_level", "")

        # Handle unparseable responses (e.g. only code, no numbered fields)
        if not intervention and not hypothesis:
            # Check if this was a restart (new skeleton)
            prev_code_path = base / f"iter_{i-1:02d}" / "generation" / f"reward_v{i-1}.py"
            curr_code_path = base / f"iter_{i:02d}" / "generation" / f"reward_v{i}.py"
            is_restart = False
            if prev_code_path.exists() and curr_code_path.exists():
                prev_code = prev_code_path.read_text(encoding="utf-8")
                curr_code = curr_code_path.read_text(encoding="utf-8")
                # Count component keys to detect skeleton change
                prev_comps = set(re.findall(r'"([a-z_]+)"\s*:', prev_code))
                curr_comps = set(re.findall(r'"([a-z_]+)"\s*:', curr_code))
                if prev_comps != curr_comps:
                    is_restart = True
                    intervention = f"骨架重建：组件从 {sorted(prev_comps)} 变为 {sorted(curr_comps)}"
                else:
                    intervention = "(响应格式异常，无法解析修改内容)"
            else:
                intervention = "(响应格式异常，无法解析修改内容)"
        expected = fields.get("expected_next_round", "")

        prev_score = scores.get(i - 1)
        curr_score = scores.get(i)
        prev_len = lengths.get(i - 1)
        curr_len = lengths.get(i)

        # Calculate delta
        if prev_score is not None and curr_score is not None:
            delta = curr_score - prev_score
            delta_str = f"{delta:+.1f}"
            if delta > 20:
                outcome = "✅ 大幅改善"
            elif delta > 0:
                outcome = "➖ 小幅改善"
            elif delta > -20:
                outcome = "➖ 小幅退步"
            else:
                outcome = "❌ 明显退步"
        else:
            delta_str = "?"
            outcome = "?"

        prev_str = f"{prev_score:.1f}" if prev_score is not None else "?"
        curr_str = f"{curr_score:.1f}" if curr_score is not None else "?"

        # Format length change
        if prev_len is not None and curr_len is not None:
            len_info = f"len: {prev_len:.0f}→{curr_len:.0f}"
        else:
            len_info = ""

        # Extract target component from intervention text
        target_comp = "?"
        # Multiple patterns found in real reflection responses:
        # "唯一目标组件 contact_bonus" / "唯一目标组件为 `contact_bonus`"
        # "唯一干预目标为 contact_bonus" / "目标是goal_proximity"
        # "修改contact_bonus的" / "修改 `contact_bonus` 组件"
        for pat in [
            r'(?:目标组件|干预目标|目标)(?:为|是)?\s*`?(\w+)`?',
            r'修改\s*`?(\w+)`?\s*(?:组件|的|近地|公式|系数)',
        ]:
            comp_match = re.search(pat, intervention)
            if comp_match:
                cand = comp_match.group(1)
                if re.match(r'^[a-z_][a-z0-9_]*$', cand):
                    target_comp = cand
                    break
        # Fallback: first backtick-quoted snake_case word
        if target_comp == "?":
            for m in re.finditer(r'`(\w+)`', intervention):
                cand = m.group(1)
                if not re.match(r'^[a-z_][a-z0-9_]*$', cand):
                    continue
                if cand in ('obs', 'action', 'next_obs', 'info', 'original_reward',
                            'training_progress', 'compute_reward', 'def', 'import',
                            'soft_factor', 'speed_sq', 'vx', 'vy', 'angle', 'angvel',
                            'max', 'abs', 'float', 'if', 'y_next', 'left_contact',
                            'right_contact', 'descent_target', 'contact_reward',
                            'true', 'false', 'none', 'range', 'len', 'min', 'sum',
                            'int', 'str', 'list', 'dict', 'bool', 'return', 'w_prox',
                            'w_vel', 'w_ang', 'w_contact', 'w_proxy', 'w_thrust',
                            'w_progress', 'w_success', 'w_soft', 'w_angle',
                            'comp_prox', 'comp_vel', 'comp_angle', 'comp_contact',
                            'comp_proxy', 'comp_thrust', 'comp_soft',
                            'progress_reward', 'contact_bonus', 'fuel_penalty',
                            'orientation_penalty', 'landing_speed_penalty',
                            'gate_x', 'gate_y', 'gate_vx', 'gate_vy',
                            'score_x', 'score_y', 'score_vx', 'score_vy',
                            'score_angle', 'score_contact', 'score_leg'):
                    continue
                target_comp = cand
                break
        # Code-diff fallback: compare component dicts across iterations
        if target_comp == "?" and i >= 3:
            prev_code_path = base / f"iter_{i-1:02d}" / "generation" / f"reward_v{i-1}.py"
            curr_code_path = base / f"iter_{i:02d}" / "generation" / f"reward_v{i}.py"
            if prev_code_path.exists() and curr_code_path.exists():
                prev_code = prev_code_path.read_text(encoding="utf-8")
                curr_code = curr_code_path.read_text(encoding="utf-8")
                prev_comps = _extract_component_blocks(prev_code)
                curr_comps = _extract_component_blocks(curr_code)
                # Find changed components
                changed = []
                all_names = set(list(prev_comps.keys()) + list(curr_comps.keys()))
                for name in all_names:
                    if prev_comps.get(name) != curr_comps.get(name):
                        changed.append(name)
                if len(changed) == 1:
                    target_comp = changed[0]
                elif len(changed) > 1:
                    target_comp = changed[0]  # pick first changed

        # Track component modification history
        if target_comp != "?":
            if target_comp not in component_mod_count:
                component_mod_count[target_comp] = []
            component_mod_count[target_comp].append((i, delta if isinstance(delta_str, str) and delta_str != "?" else None))

        # Truncate long text for readability
        intervention_short = intervention[:120] + "..." if len(intervention) > 120 else intervention

        lines.append(f"### v{i}：修改 `{target_comp}` → 分数 {prev_str} → {curr_str}（{delta_str}）{outcome}")
        lines.append(f"- **你做了什么**：{intervention_short}")
        if hypothesis:
            hyp_short = hypothesis[:150] + "..." if len(hypothesis) > 150 else hypothesis
            lines.append(f"- **你为什么这么做**：{hyp_short}")
        if len_info:
            lines.append(f"- **回合长度**：{len_info}")
        lines.append("")

        iterations_with_data += 1

    if iterations_with_data == 0:
        return ""

    # ── Component modification summary ──
    if len(component_mod_count) > 1 or (
        len(component_mod_count) == 1 and sum(len(v) for v in component_mod_count.values()) >= 3
    ):
        lines.append("## 你的修改轨迹（按组件分组）")
        lines.append("")
        for comp, history in sorted(component_mod_count.items(), key=lambda x: -len(x[1])):
            count = len(history)
            score_changes = []
            for it, delta in history:
                if delta is not None:
                    score_changes.append(f"v{it}({delta:+.0f})")
                else:
                    score_changes.append(f"v{it}(?)")
            changes_str = " → ".join(score_changes)
            lines.append(f"- **`{comp}`**：修改了 **{count} 次**，分数变化：{changes_str}")

        # Call out components modified 4+ times
        heavy_components = [c for c, h in component_mod_count.items() if len(h) >= 4]
        if heavy_components:
            lines.append("")
            for comp in heavy_components:
                lines.append(
                    f"⚠️ 你已修改 `{comp}` **{len(component_mod_count[comp])} 次**。"
                    f"在做任何涉及 `{comp}` 的修改之前，先确认你不是在重复过去的某个操作。"
                )

    lines.append("")
    lines.append("**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**")

    return "\n".join(lines)


def _extract_component_blocks(code: str):
    """Extract per-component assignment expressions from a reward function."""
    comps = {}
    m = re.search(r'components\s*=\s*\{([^}]+)\}', code, re.DOTALL)
    if not m:
        return comps
    var_to_name = {}
    for vm in re.finditer(r'"([a-z_]+)"\s*:\s*(\w+)', m.group(1)):
        var_to_name[vm.group(2)] = vm.group(1)
    code_before = code[:m.start()]
    lines_before = code_before.split('\n')
    for var, name in var_to_name.items():
        for line in reversed(lines_before):
            s = line.strip()
            if s.startswith(f"{var} =") or s.startswith(f"{var}="):
                comps[name] = s
                break
        if name not in comps:
            comps[name] = f"(var={var})"
    return comps


def _build_component_evolution(run_root, prefix, seed, current_iter):
    """Component evolution: iter_01 full code, then per-iter snapshots with score/len/best/restart."""
    try:
        base = Path(run_root) / prefix / f"seed_{seed}"
        if current_iter < 2:
            return ""

        # Read all reward versions + scores
        versions = {}
        scores = {}
        for i in range(1, current_iter):
            f = base / f"iter_{i:02d}" / "generation" / f"reward_v{i}.py"
            ts = base / f"iter_{i:02d}" / "training" / "training_summary.json"
            if f.exists():
                versions[i] = f.read_text(encoding="utf-8")
            if ts.exists():
                d = json.loads(ts.read_text(encoding="utf-8"))
                ee = d.get("external_eval", d)
                rewards = ee.get("episode_rewards", [])
                lengths = ee.get("episode_lengths", [])
                if rewards:
                    scores[i] = (sum(rewards) / len(rewards), sum(lengths) / len(lengths))

        if not versions:
            return ""

        # Determine best score
        best_score = max(v[0] for v in scores.values()) if scores else None

        # Extract components per iteration
        iter_comps = {i: _extract_component_blocks(code) for i, code in versions.items()}
        all_comps = sorted(set().union(*[set(c.keys()) for c in iter_comps.values()]))

        lines = ["# 组件演化", ""]

        # Baseline: iter_01 full code
        if 1 in versions:
            sc = scores.get(1, ("?", "?"))
            lines.append(f"## 基线 iter_01 (score={sc[0]:.1f}, len={sc[1]:.0f})")
            lines.append("```python")
            lines.append(versions[1].strip())
            lines.append("```\n")

        # Per-iteration snapshot
        prev_comps = iter_comps.get(1, {})
        for i in sorted(versions.keys()):
            if i == 1:
                continue
            comps = iter_comps[i]
            changed = [n for n in all_comps if comps.get(n) != prev_comps.get(n)]
            sc = scores.get(i, ("?", "?"))
            marks = []
            if best_score and sc[0] and sc[0] >= best_score:
                marks.append("BEST")
            # Detect restart: all component names changed vs previous
            if set(comps.keys()) != set(prev_comps.keys()):
                marks.append("RESTART")
            tag_str = f" [{', '.join(marks)}]" if marks else ""
            lines.append(f"## iter_{i:02d} (score={sc[0]:.1f}, len={sc[1]:.0f}){tag_str}")
            for name in all_comps:
                expr = comps.get(name, "(已删除)")
                tag = "修改" if name in changed else "未变"
                lines.append(f"- **{name}**: `{expr}`  [{tag}]")
            lines.append("")
            prev_comps = comps

        return "\n".join(lines)
    except Exception:
        return ""


def _build_component_delta(run_root, prefix, seed, current_iter):
    """Build before/after component comparison for the current edit.

    Reads training_summary.json from current and previous iterations,
    computes per-component deltas, and formats a diagnostic table.
    Only reports evidence — does NOT suggest what to do next.
    """
    try:
        # current_iter is the iteration being GENERATED (training hasn't run yet).
        # Compare the just-completed iteration (N-1) vs the previous one (N-2).
        if current_iter < 3:
            return ""  # need at least iter1 and iter2 training data
        prev_dir = Path(run_root) / prefix / f"seed_{seed}" / f"iter_{current_iter-2:02d}" / "training"
        curr_dir = Path(run_root) / prefix / f"seed_{seed}" / f"iter_{current_iter-1:02d}" / "training"

        prev_ts = prev_dir / "training_summary.json"
        curr_ts = curr_dir / "training_summary.json"
        if not prev_ts.exists() or not curr_ts.exists():
            return ""

        prev = json.loads(prev_ts.read_text(encoding="utf-8"))
        curr = json.loads(curr_ts.read_text(encoding="utf-8"))

        prev_comps = prev.get("component_summary", {}).get("component_stats", {})
        curr_comps = curr.get("component_summary", {}).get("component_stats", {})

        # Build union of component names (short form)
        all_names = set()
        for name in list(prev_comps.keys()) + list(curr_comps.keys()):
            short = name.replace("component.", "")
            if short not in ("generated_reward", "total_reward", "original_env_reward"):
                all_names.add(short)
        all_names = sorted(all_names)

        if not all_names:
            return ""

        lines = [
            "# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）",
            "",
            "| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]

        for name in all_names:
            p = prev_comps.get(f"component.{name}", prev_comps.get(name, {}))
            c = curr_comps.get(f"component.{name}", curr_comps.get(name, {}))
            if not p and not c:
                continue

            p_mean = float(p.get("mean", 0)) if p else None
            c_mean = float(c.get("mean", 0)) if c else None
            p_act = float(p.get("nonzero_rate", 0)) if p else None
            c_act = float(c.get("nonzero_rate", 0)) if c else None

            def fmt(v):
                if v is None: return "— (新增)"
                return f"{v:.3f}"

            def fmt_delta(new, old):
                if new is None: return "已删除"
                if old is None: return "新增"
                d = new - old
                sign = "+" if d > 0 else ""
                return f"{sign}{d:.3f}"

            def fmt_rate(v):
                if v is None: return "—"
                return f"{v:.1%}"

            def fmt_rate_delta(new, old):
                if new is None: return "—"
                if old is None: return "—"
                d = (new - old) * 100
                sign = "+" if d > 0 else ""
                return f"{sign}{d:.1f}pp"

            lines.append(
                f"| {name} | {fmt(p_mean)} | {fmt(c_mean)} | {fmt_delta(c_mean, p_mean)} | "
                f"{fmt_rate(p_act)} | {fmt_rate(c_act)} | {fmt_rate_delta(c_act, p_act)} |"
            )

        # Episode-level metrics delta
        prev_ev = prev.get("external_eval", {})
        curr_ev = curr.get("external_eval", {})
        if prev_ev and curr_ev:
            lines.extend([
                "",
                "| 回合级指标 | 编辑前 | 编辑后 | Δ |",
                "|---|---:|---:|---:|",
                f"| 开发得分 | {prev_ev.get('mean_eval_reward', 0):.2f} | {curr_ev.get('mean_eval_reward', 0):.2f} | "
                f"{curr_ev.get('mean_eval_reward', 0) - prev_ev.get('mean_eval_reward', 0):+.2f} |",
                f"| 平均回合长度 | {prev_ev.get('mean_episode_length', 0):.1f} | {curr_ev.get('mean_episode_length', 0):.1f} | "
                f"{curr_ev.get('mean_episode_length', 0) - prev_ev.get('mean_episode_length', 0):+.1f} |",
            ])
            prev_term = sum(1 for t in prev_ev.get("episode_terminated", []) if t)
            curr_term = sum(1 for t in curr_ev.get("episode_terminated", []) if t)
            n_eps = max(len(prev_ev.get("episode_terminated", [])), 1)
            lines.append(
                f"| 终止回合数 | {prev_term}/{n_eps} | {curr_term}/{n_eps} | "
                f"{curr_term - prev_term:+d} |"
            )

        return "\n".join(lines)
    except Exception:
        return ""


# ── Behavior Analyst Subagent ──────────────────────────────────────────────
# Called before the main reflection agent. It reads training feedback, memory,
# current/best reward code, and component deltas, then produces a compact
# diagnostic report that replaces the verbose component_evolution section.

def _build_analyst_user_prompt(feedback_md, memory_md, current_code, best_code,
                                cumulative_record, component_delta, prev_feedback_md,
                                prev_code, prev_score, current_score, checkpoint_data="",
                                all_historical_feedbacks=None, all_historical_codes=None):
    """Build a compact user prompt for the behavior analyst subagent.

    Includes ALL historical training feedbacks and reward codes so the analyst
    can build a complete skeleton evolution table across all iterations.
    """
    parts = []

    parts.append(f"# 本轮训练反馈 (score={current_score})\n{feedback_md}")

    if checkpoint_data:
        parts.append(f"# Checkpoint 评估数据（每10%步数的官方环境得分）\n{checkpoint_data}")

    parts.append(f"# 本轮奖励函数代码\n```python\n{current_code.strip()}\n```")

    # All historical data for building evolution table
    if all_historical_feedbacks:
        for i, (hist_fb, hist_code) in enumerate(zip(all_historical_feedbacks, all_historical_codes or [])):
            parts.append(f"# 历史 iter_{i+1:02d} 训练反馈\n{hist_fb}")
            parts.append(f"# 历史 iter_{i+1:02d} 奖励函数代码\n```python\n{hist_code.strip()}\n```")

    if cumulative_record:
        parts.append(f"# 修改自传（反思LLM过去每轮的修改决策和结果）\n{cumulative_record}")

    if memory_md:
        parts.append(f"# 历史记忆表\n{memory_md}")

    if best_code:
        parts.append(f"# 历史最佳奖励函数代码\n```python\n{best_code.strip()}\n```")

    return "\n\n".join(parts)


def _run_behavior_analyst(run_dir, reward_version, client, model, analyst_user_prompt, mock=False, reasoning_effort=None):
    """Run the behavior analyst subagent — diagnose what the policy learned.

    Returns the analyst report text, or '' on failure.
    """
    analyst_system_prompt = read_text("prompts/paper_v4/behavior_analyst_prompt.md")

    if mock:
        return ""

    try:
        out = client.chat(
            system_prompt=analyst_system_prompt,
            user_prompt=analyst_user_prompt,
            model=model,
            temperature=0.0,
            max_tokens=12000,
            reasoning_effort=reasoning_effort,
        )
        record_prompt(run_dir, "behavior_analyst", analyst_system_prompt, analyst_user_prompt)
        write_text(str(run_dir / f"behavior_analyst_output_{reward_version}.md"), out)
        return out.strip()
    except Exception as exc:
        print(f"  Behavior analyst: error — {exc}")
        return ""


def _environment_summary(environment_card_md):
    """Keep task and interface facts needed to interpret reward code.

    Sections 9-12 (expert task profile, reward roles, signal mapping, failure modes)
    are intentionally excluded from reflection. They are designed for the initial
    reward generator, not for iterative diagnosis. The reflection agent needs raw
    environment facts (1-7) and training evidence, not pre-digested design advice.
    """
    if not environment_card_md:
        return ""
    wanted = {1, 3, 4, 5, 7}
    sections = re.split(r"(?=^## \d+\.)", environment_card_md, flags=re.MULTILINE)
    selected = []
    for section in sections:
        match = re.match(r"^## (\d+)\.", section)
        if match and int(match.group(1)) in wanted:
            selected.append(section.strip())
    return "\n\n".join(selected)


def _compact_route_context(cfg, environment_card_md, expert_context_md=""):
    """Build a compact formula reference for the reflection agent.

    Extracts only the operator switching guide (§3) and key anti-patterns from
    the expert context, avoiding the full 57-line formula operator library.
    Returns ~15 lines instead of ~57.
    """
    content = expert_context_md
    if not content:
        return ""
    # Extract just the switching guide table from §3
    m3 = re.search(r"(## 3\. .*?)(?=\Z)", content, flags=re.DOTALL)
    if not m3:
        return ""
    sec3 = m3.group(1).strip()

    # Compress: keep the table rows, drop verbose descriptions
    lines = sec3.split("\n")
    compact = ["# Formula switching guide (evidence → operator)"]
    in_table = False
    for line in lines:
        if line.startswith("|"):
            compact.append(line)
            in_table = True
        elif in_table and not line.startswith("|"):
            break
    # If no table found, return the first 15 non-empty lines
    if len(compact) <= 1:
        body_lines = [l for l in lines if l.strip() and not l.startswith("#")]
        compact.extend(body_lines[:12])
    # Add key anti-pattern reminder
    compact.append("\nKey anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.")
    return "\n".join(compact)


def build_user_prompt(feedback_md, memory_md, previous_code, best_code=None, environment_card_md="", cfg=None, expert_context_md="", cumulative_record="", component_evolution="", component_delta="", is_rebuild=False, research_signal="", analyst_report="", stateless_baseline=False, component_stats_md="", checkpoint_data=""):
    """Assemble the reflection agent's user prompt — focused, no generic templates."""
    parts = []

    # ── Baseline mode: stateless — only 4 sections, no history, no analyst ──
    if stateless_baseline:
        prev_score = "?"
        m = re.search(r"score=([-\d.]+)", feedback_md)
        if m:
            prev_score = m.group(1)
        target_score = float((cfg or {}).get("iteration", {}).get("target_score", 0.0))
        current_score = float(prev_score) if prev_score != "?" else None
        if target_score > 0 and current_score is not None:
            gap = target_score - current_score
            parts.append(
                "# 1. Search objective\n"
                f"- target_score: {target_score:.6f}\n"
                f"- current_score: {current_score:.6f}\n"
                f"- gap_to_target: {gap:.6f}"
            )
        parts.append(f"# 2. Current reward program (score: {prev_score})\n```python\n{previous_code.strip()}\n```")
        parts.append(f"# 3. Training feedback\n{feedback_md}")
        if checkpoint_data:
            parts.append(f"# 3.5. Checkpoint evaluations (every 10% of training)\n{checkpoint_data}")
        env_summary = _environment_summary(environment_card_md)
        if env_summary:
            parts.append(f"# 4. Environment facts\n{env_summary}")
        return "\n\n".join(parts)

    prev_score = "?"
    m = re.search(r"score=([-\d.]+)", feedback_md)
    if m:
        prev_score = m.group(1)

    target_score = float((cfg or {}).get("iteration", {}).get("target_score", 0.0))
    current_score = float(prev_score) if prev_score != "?" else None

    if is_rebuild:
        parts.append(
            "# ⚠️ REBUILD MODE\n"
            "系统接受了你的 Level 3 重建建议。你不是在修改上一轮代码——你是在基于全部历史设计新骨架。\n"
            "参考 #2 修改自传避开已失败的路径，参考 #8 完整公式算子库选新的主信号框架。\n"
            "不要受上一轮代码结构约束。\n"
        )

    if target_score > 0 and current_score is not None:
        gap = target_score - current_score
        ratio = current_score / target_score
        parts.append(
            "# 1. Search objective\n"
            f"- target_score: {target_score:.6f}\n"
            f"- current_score: {current_score:.6f}\n"
            f"- gap_to_target: {gap:.6f}\n"
            f"- target_achievement_ratio: {ratio:.3%}"
        )

    # ── Autobiography FIRST — read your own history before looking at current code ──
    if cumulative_record:
        parts.append(cumulative_record)
    else:
        parts.append("# 2. 你的修改自传\n（这是你第一次反思，没有历史。专注于当前反馈。）")

    parts.append(f"# 3. 上一轮奖励函数代码（该轮得分: {prev_score}）\n```python\n{previous_code.strip()}\n```")

    # Behavior analyst report replaces the verbose component_evolution
    if analyst_report:
        parts.append(f"# 4. Subagent 行为诊断报告\n\n{analyst_report}")
    elif component_evolution:
        parts.append(component_evolution)

    # Best reward code — critical when current << best, so the LLM can base modifications on it
    if best_code:
        parts.append(
            "# 4.5. 历史最佳奖励函数代码\n"
            "如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，"
            "应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。\n"
            f"```python\n{best_code.strip()}\n```"
        )

    if component_delta:
        parts.append(component_delta)
    else:
        parts.append("# 5. 组件逐项对比\n（第一轮反思，无历史对比）")

    parts.append(f"# 6. 本轮训练反馈\n{feedback_md}")

    environment_summary = _environment_summary(environment_card_md)
    if environment_summary:
        parts.append(
            "# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）\n"
            f"{environment_summary}"
        )

    if is_rebuild and expert_context_md:
        parts.append(f"# 8. Formula Operator Library（完整版，用于 Level 3 重建）\n{expert_context_md}")

    if memory_md:
        parts.append(f"# 8. 历史记忆\n{memory_md}")

    return "\n\n".join(parts)


def _score_only_feedback(feedback_md):
    outcome = re.search(
        r"## Final-policy outcome\s*(.*?)(?=\n## |\Z)",
        feedback_md,
        flags=re.DOTALL,
    )
    distribution = re.search(
        r"## Evaluation distribution\s*(.*?)(?=\n## |\Z)",
        feedback_md,
        flags=re.DOTALL,
    )
    blocks = ["# Score-Only Feedback Ablation"]
    if outcome:
        blocks.extend(["## Final-policy outcome", outcome.group(1).strip()])
    if distribution:
        blocks.extend(["## Evaluation distribution", distribution.group(1).strip()])
    return "\n\n".join(blocks)


def _eureka_style_feedback(feedback_md):
    """EUREKA-style feedback: score + simple component value list, no share/rate table.

    Matches the level of detail in EUREKA's reward reflection, which tracks scalar
    values of individual reward components during training and presents them as a
    simple text summary without structured diagnostic fields (magnitude_share,
    signed_share, active_rate).
    """
    outcome = re.search(
        r"## Final-policy outcome\s*(.*?)(?=\n## |\Z)",
        feedback_md,
        flags=re.DOTALL,
    )
    # Extract component names and episode_sum_mean from the composition table
    comp_table = re.search(
        r"\| component \| episode_sum_mean.*?\n((?:\|.*?\n)+)",
        feedback_md,
        flags=re.DOTALL,
    )
    blocks = ["# Training Feedback (EUREKA-style)"]
    if outcome:
        blocks.extend(["## Final-policy outcome", outcome.group(1).strip()])
    if comp_table:
        rows = []
        for line in comp_table.group(1).strip().split("\n"):
            line = line.strip()
            if not line or "---" in line or "component" in line.lower():
                continue
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) >= 2:
                rows.append(f"- {cols[0]}: {cols[1]}")
        if rows:
            blocks.append("## Reward component values (mean per episode)\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def _score_only_memory(memory_md):
    rows = []
    for line in memory_md.splitlines():
        if not line.startswith("|") or line.startswith("|---") or "| iter |" in line:
            continue
        columns = [item.strip() for item in line.strip().strip("|").split("|")]
        if len(columns) >= 6:
            rows.append(columns[:6])
    if not rows:
        return ""
    lines = [
        "# Score-Only Reward Memory",
        "",
        "| iter | skeleton | score | best | delta | len |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _tool_definitions():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_reward_design_knowledge",
                "description": "搜索奖励设计技法库。输入症状描述（自然语言），返回匹配的技法和修复方案。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "症状的自然语言描述，如 'penalty dominating progress signal' 或 'landing bonus never triggers'",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_skeleton_detail",
                "description": "获取某个骨架的数学形态、设计原理、常见陷阱和推荐配合。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skeleton_name": {
                            "type": "string",
                            "description": "骨架名称，如 'progress_delta_reward', 'potential_based_shaping', 'bounded_proximity_reward'",
                        }
                    },
                    "required": ["skeleton_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_reward_transformation",
                "description": "Retrieve general reward-structure transformations from diagnosis evidence, including rationale, risks, and next-round verification targets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The diagnosed problem dimension or desired transformation, such as persistent proxy farming, sparse credit, product collapse, or global constraint interference.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def _run_tool_call(tool_call):
    args = json.loads(tool_call.function.arguments)
    if tool_call.function.name == "search_reward_design_knowledge":
        result = search_reward_design_knowledge(args.get("query", ""))
    elif tool_call.function.name == "get_skeleton_detail":
        result = get_skeleton_detail(args.get("skeleton_name", ""))
    elif tool_call.function.name == "get_reward_transformation":
        result = get_reward_transformation(args.get("query", ""))
    else:
        result = f"Unknown tool: {tool_call.function.name}"
    return args, result


def _parse_agent_level(response_text):
    """Extract the agent's Level decision from its response.

    Returns (level_int, level_line) where level_int is 1/2/3 or None if unparseable.
    """
    m = re.search(r"\*\*level\*\*:\s*Level\s*(\d)", response_text, re.IGNORECASE)
    if m:
        return int(m.group(1)), m.group(0).strip()
    # Fallback: check for "Level 3" anywhere in the text
    if re.search(r"Level\s*3", response_text, re.IGNORECASE):
        return 3, "Level 3 (detected from text)"
    return None, ""


def run_reflection_agent(
    config_path,
    previous_reward_path,
    best_reward_path,
    train_run_dir,
    memory_path,
    out_run_name,
    reward_version,
    environment_card_path=None,
    mock=False,
    validation_retry=None,
    duplicate_retry=None,
    is_rebuild=False,
):
    cfg = load_config(config_path)
    ablation_cfg = cfg.get("ablation", {})
    run_dir = Path(cfg["experiment"]["run_root"]) / out_run_name
    for sub in ["llm_inputs", "prompt_records", "response_records", "validations"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    reflection_mode = ablation_cfg.get("reflection_mode", "structured")
    prompt_path = ablation_cfg.get("reflection_prompt_path")
    if not prompt_path:
        prompt_path = (
            "prompts/reflection_agent_unconstrained_prompt.md"
            if reflection_mode == "unconstrained"
            else "prompts/reflection_agent_prompt.md"
        )
    system_prompt = read_text(prompt_path)
    previous_code = read_text(previous_reward_path)
    feedback_md = read_text(str(Path(train_run_dir) / "training_feedback.md"))
    if ablation_cfg.get("feedback_mode") == "score_only":
        feedback_md = _score_only_feedback(feedback_md)
    elif ablation_cfg.get("feedback_mode") == "eureka_style":
        feedback_md = _eureka_style_feedback(feedback_md)

    # ── Component checkpoint trajectories (EUREKA-style) ──
    component_stats_md = ""
    stats_path = Path(train_run_dir) / "component_stats.md"
    if stats_path.exists():
        component_stats_md = read_text(str(stats_path))

    # ── Checkpoint evaluation data (every 10% of training) ──
    checkpoint_data = ""
    ckpt_path = Path(train_run_dir) / "checkpoint_evals.json"
    if ckpt_path.exists():
        import json as _json
        try:
            ckpt_raw = _json.loads(ckpt_path.read_text())
            # Format as compact table
            lines = ["| pct | score_mean | score_std |", "|---|---|---|"]
            for r in ckpt_raw:
                pct = r.get("pct", "?")
                sm = r.get("score_mean", "?")
                ss = r.get("score_std", "?")
                if isinstance(sm, (int, float)):
                    lines.append(f"| {pct}% | {sm:.2f} | {ss:.2f} |")
            checkpoint_data = "\n".join(lines)
        except Exception:
            checkpoint_data = ""

    environment_card_md = ""
    if environment_card_path and Path(environment_card_path).exists():
        environment_card_md = read_text(environment_card_path)

    # Read the formula operator library from the same directory as the environment card
    expert_context_md = ""
    if environment_card_path:
        expert_path = Path(environment_card_path).parent / "expert_reward_context.md"
        if expert_path.exists():
            expert_context_md = read_text(str(expert_path))

    memory_md = ""
    if not ablation_cfg.get("disable_memory", False) and Path(memory_path).exists():
        memory_md = read_text(memory_path)
        if ablation_cfg.get("feedback_mode") == "score_only":
            memory_md = _score_only_memory(memory_md)
        # eureka_style keeps full memory table — only feedback is stripped

    best_code = ""
    if best_reward_path and Path(best_reward_path).exists():
        best_code = read_text(best_reward_path)

    # Generate cumulative record from all previous iterations
    cumulative_record = ""
    if not validation_retry:  # skip for pure code-fix retries
        # Extract iter number from out_run_name like "paper_ant_v6/seed_0/iter_07/generation"
        m_iter = re.search(r"/iter_(\d+)/", out_run_name)
        current_iter = int(m_iter.group(1)) if m_iter else 1
        # Extract prefix and seed from out_run_name
        # Format: {prefix}/seed_{N}/iter_{M}/generation
        parts = out_run_name.split("/")
        prefix = parts[0] if len(parts) > 0 else ""
        seed_str = ""
        for p in parts:
            if p.startswith("seed_"):
                seed_str = p.replace("seed_", "")
                break
        if prefix and seed_str and current_iter > 1:
            cumulative_record = _build_cumulative_record(
                cfg["experiment"]["run_root"], prefix, seed_str, current_iter, memory_md
            )

    # Build component evolution + numerical delta
    component_evolution = ""
    component_delta = ""
    if not validation_retry and prefix and seed_str and current_iter > 1:
        component_evolution = _build_component_evolution(
            cfg["experiment"]["run_root"], prefix, seed_str, current_iter
        )
        component_delta = _build_component_delta(
            cfg["experiment"]["run_root"], prefix, seed_str, current_iter
        )

    # ── Run subagent investigator (agentic bridge) ──
    research_signal = ""
    if not validation_retry and not duplicate_retry and not mock:
        try:
            subagent_cfg = cfg.get("subagent_investigator", {})
            if subagent_cfg.get("enabled", True):
                llm_cfg = cfg["llm"]
                client = create_client(cfg)
                result = run_investigator(
                    train_dir=str(train_run_dir),
                    previous_reward_path=str(previous_reward_path),
                    memory_path=str(memory_path) if Path(memory_path).exists() else "",
                    client=client,
                    model=llm_cfg.get("model_investigator", llm_cfg.get("model_reflection", llm_cfg["model_reward"])),
                )
                if result.get("research_signal_text"):
                    research_signal = result["research_signal_text"]
                    print(f"  Subagent: {result['turns_used']} turns, signal={len(research_signal)} chars")
                else:
                    print("  Subagent: no valid signal produced")
                # Always persist trace for audit
                trace = {
                    "turns_used": result["turns_used"],
                    "signal": result["research_signal"],
                    "tool_trace": result["tool_trace"],
                    "signal_produced": bool(result.get("research_signal_text")),
                }
                write_json(str(run_dir / f"subagent_trace_{reward_version}.json"), trace)
                if research_signal:
                    write_text(
                        str(run_dir / f"subagent_signal_{reward_version}.md"),
                        f"# Subagent Research Signal\n\n{research_signal}\n",
                    )
        except Exception as exc:
            print(f"  Subagent: error (continuing without signal) — {exc}")

    # ── Run behavior analyst (replaces component_evolution) ──
    # Skip in baseline mode: stateless reflection has no analyst, no history
    analyst_report = ""
    stateless_baseline = ablation_cfg.get("stateless_baseline", False)
    if not validation_retry and not duplicate_retry and not stateless_baseline:
        try:
            llm_cfg = cfg["llm"]
            analyst_client = create_client(cfg)

            # Collect ALL historical data for the analyst
            prev_feedback_md = ""
            prev_code = ""
            prev_score = "?"
            all_historical_feedbacks = []
            all_historical_codes = []
            if current_iter > 1 and prefix and seed_str:
                base_path = Path(cfg["experiment"]["run_root"]) / prefix / f"seed_{seed_str}"
                for i in range(1, current_iter):
                    fb_path = base_path / f"iter_{i:02d}" / "training" / "training_feedback.md"
                    code_path = base_path / f"iter_{i:02d}" / "generation" / f"reward_v{i}.py"
                    if fb_path.exists():
                        all_historical_feedbacks.append(read_text(str(fb_path)))
                    if code_path.exists():
                        all_historical_codes.append(read_text(str(code_path)))
                if all_historical_feedbacks:
                    prev_feedback_md = all_historical_feedbacks[-1]
                    prev_code = all_historical_codes[-1] if all_historical_codes else ""
                    m = re.search(r"score=([-\d.]+)", prev_feedback_md)
                    if m:
                        prev_score = m.group(1)

            current_score_str = "?"
            m_cur = re.search(r"score=([-\d.]+)", feedback_md)
            if m_cur:
                current_score_str = m_cur.group(1)

            analyst_user_prompt = _build_analyst_user_prompt(
                feedback_md, memory_md, previous_code, best_code,
                cumulative_record, component_delta,
                prev_feedback_md, prev_code, prev_score, current_score_str,
                checkpoint_data=checkpoint_data,
                all_historical_feedbacks=all_historical_feedbacks,
                all_historical_codes=all_historical_codes,
            )
            analyst_report = _run_behavior_analyst(
                run_dir, reward_version, analyst_client,
                llm_cfg.get("model_investigator", llm_cfg.get("model_reflection", llm_cfg["model_reward"])),
                analyst_user_prompt, mock,
                reasoning_effort=llm_cfg.get("reasoning_analyst"),
            )
            if analyst_report:
                print(f"  Behavior analyst: {len(analyst_report)} chars")
        except Exception as exc:
            print(f"  Behavior analyst: error (continuing without) — {exc}")

    if duplicate_retry:
        duplicate_draft_path = run_dir / f"reward_{reward_version}.py"
        duplicate_draft = read_text(duplicate_draft_path) if duplicate_draft_path.exists() else ""
        user_prompt = (
            "# Duplicate reward retry\n"
            f"{duplicate_retry}\n"
            "The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. "
            "Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. "
            "Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. "
            "Return a complete reward function whose executable code is materially different from every historical reward. "
            "Do not merely rename variables or comments.\n\n"
            f"# Rejected duplicate draft\n```python\n{duplicate_draft}\n```\n\n"
        ) + build_user_prompt(feedback_md, memory_md, previous_code, best_code, environment_card_md, cfg, expert_context_md, cumulative_record, component_evolution, component_delta, is_rebuild, research_signal, analyst_report, stateless_baseline=stateless_baseline, component_stats_md=component_stats_md, checkpoint_data=checkpoint_data)
    elif validation_retry:
        failed_draft_path = run_dir / f"reward_{reward_version}.md"
        failed_draft = read_text(failed_draft_path) if failed_draft_path.exists() else ""
        # CRITICAL: pass ALL the same context as the normal path — the LLM needs its full
        # diagnostic memory to fix the format without losing its original reasoning.
        # Only difference from normal path: the validation error header + failed draft prefix.
        user_prompt = (
            f"# ⚠️ 上一版代码验证失败\n"
            f"错误信息：{validation_retry}\n"
            "这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。"
            "直接输出修复后的完整 Python 代码。\n\n"
            f"# 被截断或无效的上一版草稿\n{failed_draft}\n\n"
        ) + build_user_prompt(
            feedback_md, memory_md, previous_code, best_code,
            environment_card_md, cfg, expert_context_md,
            cumulative_record, component_evolution, component_delta,
            is_rebuild, research_signal, analyst_report,
            stateless_baseline=stateless_baseline,
            component_stats_md=component_stats_md,
            checkpoint_data=checkpoint_data,
        )
    else:
        user_prompt = build_user_prompt(feedback_md, memory_md, previous_code, best_code, environment_card_md, cfg, expert_context_md, cumulative_record, component_evolution, component_delta, is_rebuild, research_signal, analyst_report, stateless_baseline=stateless_baseline, component_stats_md=component_stats_md, checkpoint_data=checkpoint_data)

    write_text(run_dir / f"llm_inputs/reward_{reward_version}_reflection_agent.input.md", user_prompt)
    record_prompt(run_dir, "agent_reflection", system_prompt, user_prompt)

    tool_trace = []
    trace_path = run_dir / f"response_records/reward_{reward_version}_tool_trace.json"
    previous_invocations = []
    if trace_path.exists():
        previous_trace = json.loads(read_text(trace_path))
        previous_invocations = previous_trace.get("invocations", [])
        if not previous_invocations and previous_trace.get("calls") is not None:
            previous_invocations = [{
                "validation_retry": None,
                "status": "legacy_completed",
                "calls": previous_trace.get("calls", []),
            }]

    def save_tool_trace(status):
        invocations = previous_invocations + [{
            "validation_retry": bool(validation_retry),
            "duplicate_retry": bool(duplicate_retry),
            "status": status,
            "calls": tool_trace,
        }]
        write_json(trace_path, {
            "call_count": sum(len(item.get("calls", [])) for item in invocations),
            "invocation_count": len(invocations),
            "invocations": invocations,
            "calls": [call for item in invocations for call in item.get("calls", [])],
        })

    if mock:
        from .run_05_reward_revision import MOCK_REVISION_MD
        out_md = MOCK_REVISION_MD
    else:
        llm_cfg = cfg["llm"]
        client = create_client(cfg)

        try:
            resp = client.completion(
                model=llm_cfg.get("model_reflection", llm_cfg["model_reward"]),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=llm_cfg["temperature_reward_generator"],
                max_tokens=llm_cfg["max_tokens_reward"],
                reasoning_effort=llm_cfg.get("reasoning_reflection"),
            )
        except Exception:
            save_tool_trace("llm_error")
            raise
        out_md = resp.choices[0].message.content or ""

    record_response(run_dir, "agent_reflection", out_md)
    save_tool_trace("completed")

    code = extract_code(out_md)
    validation = validate_code(code)

    write_text(run_dir / f"reward_{reward_version}.md", out_md)
    write_text(run_dir / f"reward_{reward_version}.py", code)
    write_json(run_dir / f"validations/reward_{reward_version}.validation.json", validation)

    if not validation["valid"]:
        print("WARNING: reward revision validation failed")
        for e in validation["errors"]:
            print(" -", e)
    if validation.get("warnings"):
        print("reward revision validation warnings")
        for w in validation["warnings"]:
            print(" -", w)

    print(run_dir / f"reward_{reward_version}.py")
    print(run_dir / f"reward_{reward_version}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--previous-reward", required=True)
    ap.add_argument("--environment-card", default=None)
    ap.add_argument("--best-reward", default=None)
    ap.add_argument("--train-run-dir", required=True)
    ap.add_argument("--memory", default="runs/env_001/memory/reward_memory.md")
    ap.add_argument("--out-run-name", required=True)
    ap.add_argument("--reward-version", default="v2")
    ap.add_argument("--validation-retry", default=None)
    ap.add_argument("--duplicate-retry", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    run_reflection_agent(
        config_path=args.config,
        previous_reward_path=args.previous_reward,
        environment_card_path=args.environment_card,
        best_reward_path=args.best_reward,
        train_run_dir=args.train_run_dir,
        memory_path=args.memory,
        out_run_name=args.out_run_name,
        reward_version=args.reward_version,
        mock=args.mock,
        validation_retry=args.validation_retry,
        duplicate_retry=args.duplicate_retry,
        is_rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
