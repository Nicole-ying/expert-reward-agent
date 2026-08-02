# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------- unpack observations -------------------
    x,  y  = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle      = obs[4]
    angvel     = obs[5]
    left_leg   = obs[6]
    right_leg  = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle  = next_obs[4]
    n_angvel = next_obs[5]
    n_left   = next_obs[6]
    n_right  = next_obs[7]

    # ------------------- helper quantities -------------------
    dist      = (x**2  + y**2)  ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    vel_abs       = (vx**2 + vy**2) ** 0.5
    next_vel_abs  = (nvx**2 + nvy**2) ** 0.5

    # ------------------- thresholds & weights -------------------
    w_progress = 1.0
    w_landing  = 2.0

    th_angle  = 0.5    # radians, about 30 degrees
    th_vel    = 1.0    # speed magnitude
    th_angvel = 2.0    # rad/s
    th_dist   = 0.5    # proximity to target for landing bonus

    gate_min = 0.1  # floor for each individual gate

    # ------------------- 1. progress signal (distance delta) -------------------
    # only reward moving closer, no penalty for moving away
    delta_dist = max(0.0, dist - next_dist)

    # ------------------- 2. soft health gate -------------------
    gate_angle  = max(gate_min, 1.0 - abs(angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - vel_abs      / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(angvel)  / th_angvel)
    gate = gate_angle * gate_vel * gate_angvel

    progress_gated = w_progress * delta_dist * gate

    # ------------------- 3. soft landing proxy -------------------
    # contact: at least one leg touching in the next state
    contact_next = 1.0 if (n_left + n_right) >= 1.0 else 0.0

    # stability factors after the step
    factor_angle  = max(0.0, 1.0 - abs(n_angle)  / th_angle)
    factor_vel    = max(0.0, 1.0 - next_vel_abs   / th_vel)
    factor_angvel = max(0.0, 1.0 - abs(n_angvel)  / th_angvel)
    factor_dist   = max(0.0, 1.0 - next_dist       / th_dist)

    landing_score = contact_next * factor_angle * factor_vel * factor_angvel * factor_dist
    landing_reward = w_landing * landing_score

    # ------------------- total reward -------------------
    total_reward = progress_gated + landing_reward

    components = {
        'progress_gated': progress_gated,
        'soft_landing':    landing_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **control_type**: discrete (4 actions: no engine, left orientation, main vertical, right orientation)

## selected reward roles
根据 environment_card 的任务画像与角色预算，v1 选择以下三个组件角色：
1. **主学习信号（progress_reward）**：使用距离改善量 `delta_dist`，鼓励 agent 每步靠近目标着陆点。
2. **健康/稳定门控（soft_health_gate）**：将姿态倾角、速度大小、角速度转换为乘性因子，在主奖励上衰减——当身体状态恶化时抑制进展信号，引导 agent 先保持姿态再移动。
3. **任务完成近似信号（soft_landing_proxy）**：当下一步支撑腿接触且姿态、速度、角速度、距目标距离均接近理想值时，给予额外正奖励，为软着陆提供明确的完成引导。

## role_to_signal_mapping
- **progress_reward** → `obs[0], obs[1], next_obs[0], next_obs[1]`（位置，计算欧氏距离差值）
- **soft_health_gate** → `obs[2], obs[3]`（速度）、`obs[4]`（姿态角）、`obs[5]`（角速度），线性衰减门
- **soft_landing_proxy** → `next_obs[6], next_obs[7]`（支撑腿接触）、`next_obs[0], next_obs[1]`（下一时刻位置）、`next_obs[2], next_obs[3], next_obs[4], next_obs[5]`（下一时刻速度与姿态）

## 每个 role 选择的 formula operator
- **progress**：`improvement_delta`，形式 `max(0, dist - next_dist)`
- **health gate**：三个独立的 `bounded_signal` 门（线性衰减 `1 - |error|/threshold`），相乘得到 gate，再乘到 progress 上，等效于 `soft_health_gate`
- **soft landing**：`joint_condition_proxy` 的乘积形式 `contact * factor_angle * factor_vel * factor_angvel * factor_dist`，每个因子均为 `max(0, 1 - ratio)` 的连续 bounded 信号

## excluded roles 及原因
- **terminal_success_reward / terminal_failure_penalty**：info 为空，无显式 success/failure flag；且环境终止时 `next_obs` 可能已是重置后的初始状态，不可靠，故未使用。
- **action_efficiency**：离散动作空间下暂不引入动作代价，v1 重心在学习安全降落路径；能耗优化留到后续迭代。
- **strong gated reward / dynamic curriculum**：本任务无明确阶段划分，v1 不需要复杂门控或训练进度依赖。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- 环境卡片明确 `explicit_success_flag_available=false`，`info` 为空字典，没有任何成功/失败标志可读取。
- `compute_reward` 接口未提供 `terminated` 参数，无法可靠判断当前步是否为最后一步。
- 即使通过观测推断（如位置出界），环境终止后 next_obs 通常为重置状态，推断出的 “崩溃” 信号可能污染奖励；因此 v1 完全避免 terminal event。

## 哪些职责留到后续迭代
- **动作效率/燃料消耗**：未来可加入离散动作代价（非 no_engine 动作轻微惩罚）以鼓励最少引擎使用，但当前阶段先确保平稳着陆。
- **动态难度/课程**：若发现 agent 在随机初始扰动下学习困难，可逐步引入 based-shaping 或潜在课程，但在基础不确定性处理之前不增加。
- **更精细的着陆条件**：目前 soft_landing 使用乘积；若观察到某条件常导致乘积塌缩（例如角速度 never 同时满足），可改为几何平均 `(factor1 * factor2 * ...) ** (1/n)`。
- **预览信息**：当前观测无 “前方” 地形信息，无需 `preview_conditioned_reward`。

## 训练后应该观察哪些 failure modes
1. **悬停不降**：agent 为了保持 gate 不衰减，可能选择悬停在空中而不下降，此时 `progress_gated` 接近零。需检查阈值是否过严，或考虑在后续加入高度相关的增量奖励。
2. **冲撞地面**：agent 快速下降但倾角与速度增大，gate 衰减但 `delta_dist` 依然正，导致 reward 总和仍高，最终 crash。需要观察终止时的姿态/速度分布，必要时收紧 `th_vel`、`th_angle`。
3. **在错误位置软着陆**：`soft_landing` 的 `factor_dist` 可抑制远离目标的着陆奖励，但若 `th_dist=0.5` 仍过大，agent 可能在平台边缘或沟壑中软着陆并持续获得奖励。需监控着陆时 `x,y` 的位置分布。
4. **反复微调引擎靠门控刷分**：gate 的 `gate_min=0.1` 可能导致 agent 通过高频小摆幅维持 gate 在低位但仍获得少量 progress，累计 reward 可支撑存活。必要时可降低 gate_min 或提升 progress 权重。