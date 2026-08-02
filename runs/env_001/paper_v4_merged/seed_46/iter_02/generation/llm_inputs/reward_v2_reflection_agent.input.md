# 1. Search objective
- target_score: 200.000000
- current_score: 236.610458
- gap_to_target: -36.610458
- target_achievement_ratio: 118.305%

# 2. 上一轮奖励函数代码（该轮得分: 236.610458）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    # obs: [x_pos, y_pos, x_vel, y_vel, body_angle, angular_vel, left_contact, right_contact]
    ox, oy, ovx, ovy, oangle, oav, olc, orc = obs
    nx, ny, nvx, nvy, nangle, nav, nlc, nrc = next_obs

    # Compute distances
    old_dist = (ox * ox + oy * oy) ** 0.5
    new_dist = (nx * nx + ny * ny) ** 0.5

    # --- Goal proximity progress (main learning signal) ---
    raw_progress = old_dist - new_dist   # positive when moving closer
    max_delta = 5.0                      # clip single-step changes
    progress = max(-max_delta, min(max_delta, raw_progress))
    progress_reward = 1.0 * progress     # weight = 1.0

    # --- Landing gentleness (constraint) ---
    CLOSE_DIST = 3.0
    SAFE_SPEED = 1.0
    LAND_WEIGHT = 0.5
    speed = (nvx * nvx + nvy * nvy) ** 0.5
    # Linear activation inside the close region (0 -> 1 as distance decreases)
    close_factor = max(0.0, 1.0 - new_dist / CLOSE_DIST)
    # Hinge penalty on excess speed, scaled by close_factor
    landing_penalty = -LAND_WEIGHT * max(0.0, speed - SAFE_SPEED) * close_factor

    # --- Orientation penalty (constraint) ---
    ANGLE_THRESHOLD = 0.3   # radians
    ORIENT_WEIGHT = 0.2
    orientation_penalty = -ORIENT_WEIGHT * max(0.0, abs(nangle) - ANGLE_THRESHOLD)

    # --- Terminal success bonus (task-completion proxy) ---
    SUCCESS_DIST = 0.2
    SUCCESS_SPEED = 0.5
    SUCCESS_ANGLE = 0.2
    SUCCESS_BONUS = 0.2
    success_bonus = 0.0
    if (new_dist < SUCCESS_DIST and speed < SUCCESS_SPEED
            and abs(nangle) < SUCCESS_ANGLE
            and (nlc > 0.5 or nrc > 0.5)):
        success_bonus = SUCCESS_BONUS

    total_reward = progress_reward + landing_penalty + orientation_penalty + success_bonus

    components = {
        "goal_proximity_progress": progress_reward,
        "landing_gentleness_penalty": landing_penalty,
        "orientation_penalty": orientation_penalty,
        "terminal_success_bonus": success_bonus
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=236.610458, len=419.850000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[64.846763, 310.187661]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success_bonus | 40.220000 | 96.3% | 96.3% | 47.9% |
| goal_proximity_progress | 1.326154 | 3.2% | 3.4% | 97.8% |
| orientation_penalty | -0.099911 | -0.2% | 0.2% | 2.3% |
| landing_gentleness_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Score 236.6, termination 85%, episode length ~420. Terminal_success_bonus dominates with 96.3% signed share.

**Component Anomalies**: landing_gentleness_penalty dead (active 0%). orientation_penalty near-dead (active 2.3%, -0.2% share). terminal_success_bonus >70% share.

**Training Dynamics**: No temporal snapshots provided; dynamics over checkpoints unavailable.

**Signal Quality**: Dead gate landing_gentleness. Progress reward active (97.8%) but contributes only 3.2% share. Success bonus fires in 47.9% steps, driving high accumulation.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器从画面顶部附近出发，尽快飞到中央目标平台并稳定着陆，同时尽量减少引擎推力使用。  
次优要求：保持姿态稳定、安全轻触平台、避免侧向偏移过大。  
不应混淆的目标：单纯存活（有明确的到达位置要求），也不是无限制前进（目标是一个固定点位）。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `(8,)`
- dtype: `float32`（假定，实际由环境决定）
- obs[0]: `x_position` – 相对于目标平台的水平距离，`reward_usable: true`
- obs[1]: `y_position` – 相对于平台高度的垂直距离，`reward_usable: true`
- obs[2]: `x_velocity` – 水平线速度，`reward_usable: true`
- obs[3]: `y_velocity` – 垂直线速度，`reward_usable: true`
- obs[4]: `body_angle` – 机体倾角，`reward_usable: true`
- obs[5]: `angular_velocity` – 角速度，`reward_usable: true`
- obs[6]: `left_support_contact` – 左支撑触地标志（0/1 或连续），`reward_usable: true`
- obs[7]: `right_support_contact` – 右支撑触地标志，`reward_usable: true`

## 4. 动作空间 action_space
- type: `Discrete`
- n: `4`
- action 0: `no_engine` – 无推力，惯性飞行
- action 1: `left_orientation_engine` – 启动左姿态发动机（主要用于调整角速度）
- action 2: `main_engine` – 启动主发动机（提供反推力/升力）
- action 3: `right_orientation_engine` – 启动右姿态发动机（与左姿态对称）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` – 当机体静止（可能已着陆并稳定）时触发。此模式极可能表示成功着陆，尤其是配合近距离、低速、良好姿态和接触信号。
- **failure-like termination**: `crash_or_body_contact` – 机体与地面或其他物体非腿部接触（推测会导致姿态破坏、超出限制等）。  
  `horizontal_position_outside_viewport` – 水平位置超出视野，必然失败（飞离目标区）。
- **ambiguous termination**: 当同时满足多个条件时（如 crash 且出界），仍视为失败；但仅靠观测无法区分触发原因。`crash_or_body_contact` 和 `body_not_awake_or_settled` 可能同时触发，需以 fail 为准（因为 crash 优先级高）。
- **truncation**: 源码中未看到最大步数截断，但实际使用中可能存在。无额外截断标志。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: `false` (info 为空)
- `explicit_failure_flag_available`: `false`
- `allowed_info_fields`: 无（info 为 `{}`）
- `forbidden_or_uncertain_info_fields`: 任何假设的 `info["success"]`、`info["termination_reason"]` 等均禁止使用

即使没有明确标志，成功可通过终止后状态间接推断：位置接近零、速度极小、姿态平直且至少有一只脚接触平台。这属于 `derived_possible` 信号，在奖励设计时必须谨慎。

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]` (x 偏差), `next_obs[1]` (y 偏差) → 可计算到目标的距离
- **velocity**: `next_obs[2]`, `next_obs[3]` → 总速度或分速度
- **orientation**: `next_obs[4]` (机体倾角), `next_obs[5]` (角速度)
- **contact**: `next_obs[6]`, `next_obs[7]` – 左右支撑是否触地
- **action/engine**: 动作选择本身（离散 0‑3）可用于惩罚引擎使用
- **other**:
  - 距离变化量：`delta_distance = distance(obs) - distance(next_obs)`（进步信号）
  - 终端推断成功（derived_possible）：在 episode 结束时，结合位置、速度、倾角、接触判断是否为成功着陆，可给予稀疏终端奖励
  - 终端推断失败（derived_possible）：推断 crash/出界，可给予惩罚（谨慎使用）

# 7. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | goal_proximity_progress + landing_gentleness_penalty + orientation_penalty + terminal_success_bonus | 236.61 | 236.61 | 0.00 | 419.85 | goal_proximity_progress=0.002 landing_gentleness_penalty=-0.001 orientation_penalty=-0.002 terminal_success_bonus=0.127 | target_solved_new_best |
