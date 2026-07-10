# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度 + 稳定性门控
    hull_angle = obs[0]                  # 身体倾斜角
    horizontal_velocity = obs[2]         # 前向速度
    k_angle = 1.0
    stability = 1.0 / (1.0 + k_angle * abs(hull_angle))
    forward_vel = max(horizontal_velocity, 0.0)
    w_progress = 1.0
    forward_progress = w_progress * stability * forward_vel

    # 姿态惩罚：鼓励保持直立
    w_posture = 0.1
    posture_penalty = -w_posture * (hull_angle ** 2)

    # 角速度惩罚：抑制剧烈旋转
    hull_angular_velocity = obs[1]
    w_angvel = 0.01
    angvel_penalty = -w_angvel * (hull_angular_velocity ** 2)

    # 动作效率惩罚：极小权重降低能耗
    w_action = 0.001
    action_sum_sq = sum(a**2 for a in action)
    action_penalty = -w_action * action_sum_sq

    total_reward = forward_progress + posture_penalty + angvel_penalty + action_penalty

    components = {
        "forward_progress": forward_progress,
        "posture_penalty": posture_penalty,
        "angular_vel_penalty": angvel_penalty,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  `locomotion_continuous_control`, subtype `planar_bipedal_gait`。任务是驱动双足模型在崎岖地形上尽可能快、远地前进，同时降低能耗。

- **selected reward roles**  
  1. **主学习信号** – 前进速度 × 稳定性门控（`forward_progress`）。  
  2. **稳定/安全约束** – 身体倾斜角惩罚（`posture_penalty`）和角速度惩罚（`angular_vel_penalty`）。  
  3. **效率/动作代价** – 动作平方和惩罚（`action_penalty`），极轻量级，避免压制探索。

- **role‑to‑signal mapping**  
  - `forward_progress`: `hull_angle` (obs[0]) → 稳定性门控，`horizontal_velocity` (obs[2]) → 正向速度奖励。  
  - `posture_penalty`: `hull_angle` (obs[0]) → 二次惩罚。  
  - `angular_vel_penalty`: `hull_angular_velocity` (obs[1]) → 二次惩罚。  
  - `action_penalty`: `action` 向量 (4 维) → 平方和。

- **formula operators used**  
  - **soft_health_gate** 变形：`stability = 1/(1 + k·|hull_angle|)`，为主进度信号提供姿态门控，使不稳定步伐的奖励衰减。  
  - **dense_state_signal** (positive): 用 `max(horizontal_velocity, 0)` 只奖励前进，不惩罚后退。  
  - **quadratic_penalty**: 用于姿态、角速度、动作惩罚。  
  整体遵循“先给任务方向，再加安全约束，最后才考虑效率”的顺序。

- **excluded roles & reasons**  
  - `terminal_success_reward` / `terminal_failure_penalty`: 环境 `info` 为空，无任何显式成功/失败标志，不可构造。  
  - `task_completion_proxy`: 缺少地形终点距离或剩余步数信号，无法定义可信的软完成条件。  
  - `vertical_velocity_penalty`: 为避免组件数量膨胀，v1 暂不引入，防止过度约束。  
  - `curriculum_weighting`: v1 无需调度阶段权重，后续可根据训练进度动态调整。  
  - `original_reward`: 设计规范明确禁止使用原始环境奖励。

- **为何没有 terminal_success / failure 惩罚**  
  跌倒或到达终点都会直接终止 episode，环境自身的折扣未来奖励已间接体现“活得更久、走得更远”的激励；额外惩罚/奖励若没有可靠标志，反而引入不可靠信号。

- **留到后续迭代的职责**  
  - 利用激光雷达（obs[14:23]）构建地形感知奖励（如避障、提前规划落脚点）。  
  - 引入更细腻的步态约束（如交替接触模式、离地时间）以改进行走自然度。  
  - 动态权重课程（随着训练进度逐步增加速度要求或降低动作惩罚）。  
  - 垂直速度约束（若后期发现跳跃刷分现象）。  

- **训练后应重点观察的 failure modes**  
  - **站立不动**：若姿态/角速度惩罚过强，agent 可能选择完全静止以获得零惩罚，而不尝试前进。  
  - **短爆冲刺后摔倒**：稳定性门控未能充分抑制大角速度时的快速移动，导致 agent 仅向前冲几步就翻倒。  
  - **原地踏步但无净位移**：门控+前进奖励可能被利用于高频踏步而实际水平速度很低，需检查 horizontal_velocity 的实际值与 gait 质量。  
  - 监控指标：前进速度均值、躯干倾角标准差、存活步数分布、总奖励中各组件的比例。
