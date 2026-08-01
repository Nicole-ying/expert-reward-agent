import argparse
from pathlib import Path
from .common import load_config, read_text, write_text, make_run_dir, record_prompt, record_response
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


def run(config_path, run_name, mock=False):
    cfg = load_config(config_path)
    run_dir = make_run_dir(cfg, run_name)
    system_prompt = read_text(cfg["prompts"]["environment_analyzer"])
    task_spec = read_text(cfg["inputs"]["task_spec_path"])
    masked_step = read_text(cfg["inputs"]["masked_step_path"])
    user_prompt = f"ANONYMIZED_TASK_SPEC:\n{task_spec}\n\nMASKED_STEP_SOURCE:\n{masked_step}"
    record_prompt(run_dir, "01_environment_analyzer", system_prompt, user_prompt)

    if mock:
        env_md = MOCK_ENV_MD
    else:
        llm_cfg = cfg["llm"]
        client = DeepSeekClient(api_key_env=llm_cfg["api_key_env"], base_url=llm_cfg["base_url"])
        env_md = client.chat(
            model=llm_cfg["model_env"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=llm_cfg["temperature_environment_analyzer"],
            max_tokens=llm_cfg["max_tokens_env"],
            json_mode=False,
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
