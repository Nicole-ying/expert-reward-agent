import argparse
import re
from pathlib import Path
import yaml

from .common import load_config, read_text, write_text, write_json, make_run_dir, record_prompt, record_response
from llm_clients.deepseek_client import DeepSeekClient

MOCK_ENV_MD = """# Env_001 环境理解卡片

## 1. 任务目标
这是一个二维车辆/飞行器式控制任务。智能体需要接近中央目标区域，并尽量稳定地停在目标附近。次要目标包括减少发动机使用和更高效完成任务。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目标是到达并稳定接近目标区域，属于导航/到达目标类任务；稳定、接触和燃料是附加约束，不改变主要任务类型。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32
- obs[0]: x_position，水平位置，相对目标区域
- obs[1]: y_position，垂直位置/高度，相对目标区域
- obs[2]: x_velocity，水平速度
- obs[3]: y_velocity，垂直速度
- obs[4]: body_angle，机体角度
- obs[5]: angular_velocity，角速度
- obs[6]: left_contact，左支撑接触标志，0/1
- obs[7]: right_contact，右支撑接触标志，0/1

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine，不喷气
- action 1: left_orientation_engine，左/侧向姿态发动机
- action 2: main_engine，主发动机
- action 3: right_orientation_engine，右/侧向姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 可能存在“稳定/停止活动”类终止，但没有明确 success flag 传入 reward 函数。
- failure-like termination: 可能包括碰撞、越界、机体接触等，但没有明确 failure flag 传入 reward 函数。
- ambiguous termination: done/terminated 只有二值终止时，不能直接判断成功还是失败。
- truncation: 如果存在时间截断，也不能当作成功或失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []
- forbidden_or_uncertain_info_fields: success, failure, termination_reason, official_reward, original_reward

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- next_obs
- action
- training_progress 只有明确需要课程奖励时才用

禁止使用：
- original_reward
- official_reward
- info["success"] / info.get("success")
- info["failure"] / info.get("failure")
- info["termination_reason"]
- 未声明的 obs 切片，例如 obs[0:3]

## 7. 可用于奖励函数的信号
- position: obs[0], obs[1], next_obs[0], next_obs[1]
- velocity: obs[2], obs[3], next_obs[2], next_obs[3]
- orientation: obs[4], obs[5], next_obs[4], next_obs[5]
- contact: obs[6], obs[7], next_obs[6], next_obs[7]
- action/engine: action 可以反映是否使用发动机，但能耗项建议后续迭代再加

## 8. 不确定或不可用的信号
- explicit success flag
- explicit failure flag
- termination reason
- official reward
"""


def _task_description(task_spec):
    parsed = yaml.safe_load(task_spec) or {}
    description = parsed.get("task_description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("task_spec must contain a non-empty task_description")
    return description.strip()


def _episode_step_limit(train_cfg):
    configured = train_cfg.get("episode_step_limit")
    if configured is not None:
        return int(configured)
    try:
        import gymnasium as gym
        return int(gym.spec(train_cfg["runner_env_id"]).max_episode_steps)
    except Exception:
        return None


def _environment_user_prompt(task_spec, masked_step, reward_clip, episode_step_limit):
    clip_text = "disabled" if reward_clip is None else f"[-{float(reward_clip)}, +{float(reward_clip)}]"
    limit_text = "unknown" if episode_step_limit is None else str(int(episode_step_limit))
    return f"""# ANONYMIZED_TASK_SPEC

{task_spec.strip()}

# MASKED_STEP_SOURCE

```python
{masked_step.strip()}
```

# REWARD_INTERFACE_CONTRACT

```text
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):

Runtime-accessible inputs:
- obs, action, next_obs, training_progress
- info[\"terminated\"]: bool
- info[\"truncated\"]: bool
- info[\"done\"]: bool(terminated or truncated)

Runtime episode-limit semantics:
- an outer TimeLimit wrapper may set info[\"truncated\"]=True
- truncation means budget exhaustion, not automatic success or failure

Runtime total-reward clipping after compute_reward returns: {clip_text}
Configured maximum episode steps: {limit_text}

Forbidden:
- original_reward and official/native reward
- bare terminated, truncated, or done variables
- undeclared info fields

The termination boolean does not expose its cause. Legal state evidence may support a calibrated
heuristic, but it is not automatically a ground-truth success/failure label. The environment card
must assign an operational reliability level and permitted reward use to every terminal decision.
```
"""


def _compose_environment_card(task_description, response):
    analysis = re.sub(
        r"\A#\s+(?:Environment Semantics Card|Environment Analysis)\s*\n+",
        "",
        response.strip(),
        count=1,
        flags=re.IGNORECASE,
    )
    return (
        "# Environment Semantics Card\n\n"
        "## 0. Original anonymized task description\n\n"
        f"{task_description}\n\n{analysis.strip()}\n"
    )


def _terminal_boundary_semantic_violations(response):
    match = re.search(
        r"(?mis)^###\s+Operational terminal decision boundary\s*(.*?)(?=^###\s|^##\s|\Z)",
        response,
    )
    if not match:
        return ["operational terminal boundary is missing"]
    violations = []
    for line in match.group(1).splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip().strip("`").lower() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[0] == "decision/evidence" or "heuristic_only" not in cells[2]:
            continue
        permitted_use = cells[3]
        if any(token in permitted_use for token in ("bonus", "terminal", "binary", "event", "constant")):
            violations.append(
                f"heuristic_only row `{cells[0]}` permits binary/terminal use: {cells[3]}"
            )
        if not any(token in permitted_use for token in ("continuous", "diagnostic", "诊断", "连续")):
            violations.append(
                f"heuristic_only row `{cells[0]}` does not restrict use to continuous shaping or diagnostics"
            )
    return violations


def run(config_path, run_name, mock=False):
    cfg = load_config(config_path)
    run_dir = make_run_dir(cfg, run_name)
    system_prompt = read_text(cfg["prompts"]["environment_analyzer"])
    task_spec = read_text(cfg["inputs"]["task_spec_path"])
    masked_step = read_text(cfg["inputs"]["masked_step_path"])
    train_cfg = cfg.get("training", {})
    user_prompt = _environment_user_prompt(
        task_spec,
        masked_step,
        train_cfg.get("reward_clip", 20.0),
        _episode_step_limit(train_cfg),
    )
    record_prompt(run_dir, "01_environment_analyzer", system_prompt, user_prompt)

    if mock:
        env_md = MOCK_ENV_MD
    else:
        llm_cfg = cfg["llm"]
        client = DeepSeekClient(api_key_env=llm_cfg["api_key_env"], base_url=llm_cfg["base_url"])
        env_md = None
        for attempt in range(1, 4):
            attempt_user_prompt = user_prompt
            if attempt > 1:
                attempt_user_prompt += (
                    "\n\n# COMPLETENESS RETRY\n"
                    "The previous response was incomplete. Regenerate the full card from sections 1 through 8, "
                    "including the exact heading `### Operational terminal decision boundary`. Keep every field "
                    "compact so sections 7 and 8 are not truncated. Do not relax any evidence boundary. "
                    "For every heuristic_only row, permitted reward use must be diagnostics or continuous state "
                    "shaping only; never permit a terminal/binary bonus, penalty, event, or constant."
                )
                record_prompt(
                    run_dir,
                    f"01_environment_analyzer_attempt_{attempt}",
                    system_prompt,
                    attempt_user_prompt,
                )
            response = client.chat(
                model=llm_cfg["model_env"],
                system_prompt=system_prompt,
                user_prompt=attempt_user_prompt,
                temperature=llm_cfg["temperature_environment_analyzer"],
                max_tokens=llm_cfg["max_tokens_env"],
                json_mode=False,
            )
            record_response(run_dir, f"01_environment_analyzer_attempt_{attempt}", response)
            missing = [index for index in range(1, 9) if not re.search(rf"(?m)^## {index}\.\s", response)]
            missing_terminal_boundary = not re.search(
                r"(?mi)^###\s+Operational terminal decision boundary\s*$", response
            )
            terminal_semantic_violations = _terminal_boundary_semantic_violations(response)
            complete = not missing and not missing_terminal_boundary and not terminal_semantic_violations
            write_json(run_dir / "validations/environment_card.validation.json", {
                "complete": complete,
                "attempt": attempt,
                "missing_sections": missing,
                "missing_operational_terminal_boundary": missing_terminal_boundary,
                "terminal_semantic_violations": terminal_semantic_violations,
            })
            if complete:
                env_md = _compose_environment_card(_task_description(task_spec), response)
                break
        if env_md is None:
            raise RuntimeError(
                "Environment card remained incomplete after 3 attempts; "
                f"missing sections: {missing}, "
                f"missing operational terminal boundary: {missing_terminal_boundary}, "
                f"terminal semantic violations: {terminal_semantic_violations}"
            )

    write_text(run_dir / "environment_card.md", env_md)
    record_response(run_dir, "01_environment_analyzer", env_md)
    print(run_dir / "environment_card.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--run-name", default="mock_run")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    run(args.config, args.run_name, args.mock)
