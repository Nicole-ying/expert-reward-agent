# 1. Search objective
- target_score: 200.000000
- current_score: -10.715175
- gap_to_target: 210.715175
- target_achievement_ratio: -5.358%

# 2. 上一轮奖励函数代码（该轮得分: -10.715175）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Proximity reward + attitude-only penalty + soft landing gate.
    Removed velocity penalty that was preventing the agent from getting close
    enough to trigger soft_landing.
    """

    # ── Unpack observations ──────────────────────────────────────────
    px1, py1 = next_obs[0], next_obs[1]  # position
    vx1, vy1 = next_obs[2], next_obs[3]  # velocity (unused now, kept for clarity)
    angle1  = next_obs[4]                # body angle
    angvel1 = next_obs[5]                # angular velocity
    left_leg  = next_obs[6]              # left contact
    right_leg = next_obs[7]              # right contact

    # ── Derived signals ─────────────────────────────────────────────
    dist_next = (px1**2 + py1**2) ** 0.5
    speed = (vx1**2 + vy1**2) ** 0.5    # only used in soft_landing now

    # ── 1. Proximity reward ──────────────────────────────────────────
    proximity_reward = 1.0 / (1.0 + dist_next)

    # ── 2. Attitude penalty (replaces stability_penalty) ─────────────
    # Only penalize tilt and spinning — no speed penalty.
    angle_penalty  = -0.003 * (angle1 ** 2)
    angvel_penalty = -0.001 * (angvel1 ** 2)
    attitude_penalty = angle_penalty + angvel_penalty

    # ── 3. Soft landing guidance (unchanged) ─────────────────────────
    proximity_threshold = 0.3
    if dist_next < proximity_threshold:
        contact_factor = (left_leg + right_leg) / 2.0
        speed_factor   = 1.0 / (1.0 + 10.0 * speed)
        angle_factor   = 1.0 / (1.0 + 5.0 * (angle1**2))
        soft_landing   = contact_factor * speed_factor * angle_factor
    else:
        soft_landing = 0.0

    # ── Combine ──────────────────────────────────────────────────────
    total_reward = (
        1.0 * proximity_reward
        + 1.0 * attitude_penalty
        + 2.0 * soft_landing
    )

    components = {
        "proximity_reward":   proximity_reward,
        "attitude_penalty":   attitude_penalty,
        "soft_landing":       soft_landing,
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 436.40 | 163.33 | ✅ |
| 2 | 二次项使大步前进的奖励显著更高，agent 将更早到达 proximity 区域并仍由 `soft_landing... | 二次项使大步前进的奖励显著更高，agent 将更早到达 proximity 区域并仍由 `soft_landing... | 95.85 | -30.30 | ❌ |
| 3 | 骨架变化: orientation_penalty + safe_progress + soft_landing | — | 1000.00 | -11.06 | ❌ |
| 4 | 降低 penalty 系数 10 倍后，agent 将敢于倾斜和转动，从而有效接近目标，触发 soft_landi... | 降低 penalty 系数 10 倍后，agent 将敢于倾斜和转动，从而有效接近目标，触发 soft_landi... | 812.95 | 146.42 | ➖ |
| 5 | 密集的 distance progress + alignment 将使每步都有引导，agent 会更快趋近目标，... | 密集的 distance progress + alignment 将使每步都有引导，agent 会更快趋近目标，... | 1000.00 | -23.46 | ❌ |
| 6 | 直接奖励距离缩小（1/(1+dist)）在远处提供温和引导，在近处提供强力吸引，迫使 agent 向原点运动，最终... | 直接奖励距离缩小（1/(1+dist)）在远处提供温和引导，在近处提供强力吸引，迫使 agent 向原点运动，最终... | 630.20 | 194.63 | ✅ |
| 7 | `speed * 1/(1+dist)` 惩罚将在近距离自然压低速度，延长 soft_landing 有效窗口，从... | `speed * 1/(1+dist)` 惩罚将在近距离自然压低速度，延长 soft_landing 有效窗口，从... | 1000.00 | -10.97 | ❌ |
| 8 | 移除速度约束后，agent 将像 iter 6 一样敢于接近目标，触发 `soft_landing` 并恢复 19... | 移除速度约束后，agent 将像 iter 6 一样敢于接近目标，触发 `soft_landing` 并恢复 19... | 1000.00 | -10.72 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-10.715175, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-42.312079, 27.207855]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 801.946142 | 100.0% | 100.0% | 100.0% |
| attitude_penalty | -0.011438 | -0.0% | 0.0% | 100.0% |
| soft_landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Eval score -10.7, 0/20 terminated, all truncated at 1000 steps. Soft_landing dead (active_rate=0%, sum=0). Proximity_reward dominates.

**Component Anomalies**: Soft_landing dead in final eval despite 39.8% nonzero in train summary. Proximity_reward 100% share. Attitude_penalty negligible (-0.011 sum).

**Training Dynamics**: No temporal snapshots; cannot assess drift or plateau.

**Signal Quality**: Soft_landing threshold (dist<0.3) not regularly crossed; avg dist ~0.39. Reward uncorrelated with task (orig_env_reward -0.09). No effective attitude signal.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个 2D 飞行器/车辆类似的任务：智能体从视口顶部中心附近受初始随机力开始，需要尽可能快地飞抵并稳定停靠在中央目标平台上，同时尽量减少发动机推力使用。核心目标是 **快速、稳定地完成着陆（到达并停留）**，次要目标是 **节省燃料、保持姿态平稳**。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (推断)
- obs[0]: `x_position` – 相对于目标平台中心的水平坐标，越小越接近，reward_usable: true
- obs[1]: `y_position` – 相对于目标平台高度的垂直坐标，reward_usable: true
- obs[2]: `x_velocity` – 水平线速度，reward_usable: true
- obs[3]: `y_velocity` – 垂直线速度，reward_usable: true
- obs[4]: `body_angle` – 身体倾斜角度（假设水平为 0），reward_usable: true
- obs[5]: `angular_velocity` – 角速度，reward_usable: true
- obs[6]: `left_support_contact` – 左支撑腿是否接触（0 或 1），reward_usable: true
- obs[7]: `right_support_contact` – 右支撑腿是否接触（0 或 1），reward_usable: true

（注意：所有字段均可用，但需小心接触信号的语义，任务目标中“接触”指的是安全着陆在目标平台，而非与地面或障碍物的碰撞）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` – 不点火，依靠当前动量漂移
- action 1: `left_orientation_engine` – 点燃左姿态发动机（调整姿态或水平推力）
- action 2: `main_engine` – 点燃主发动机（主要提供垂直或前进推力）
- action 3: `right_orientation_engine` – 点燃右姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
根据掩码源码，存在三种终止触发：
- `crash_or_body_contact` – 坠毁或部分身体接触（可能包括与地面/障碍物的不当接触）
- `horizontal_position_outside_viewport` – 水平位置超出视口边界（失败）
- `body_not_awake_or_settled` – 身体不再活跃或已经稳定（可能为成功，若发生在目标平台上）

成功意义上的终止并没有显式分离，只能通过观测状态间接判别：当智能体接近目标( x ≈ 0, y ≈ 0 )，速度极小，且两侧支撑腿接触（可能），触发 `body_not_awake_or_settled` 可视为 soft landing success；而其他终止条件（crash、出界）则对应失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空字典，无额外字段）
- forbidden_or_uncertain_info_fields: 所有未在 `observation_space` 中列出的字段均不可用（包括 `terminated` 标记、`success` 等）

## 7. 可用于奖励函数的信号
- **位置**：`x_position`，`y_position`（相对目标，直接表征进度）
- **速度**：`x_velocity`，`y_velocity`（绝对值或矢量和可用于判断稳定、能耗）
- **姿态**：`body_angle`，`angular_velocity`（衡量晃动，违反稳定着陆）
- **接触**：`left_support_contact`，`right_support_contact`（区分接触/非接触，可用于软着陆推断）
- **动作/引擎**：动作类别 0-3，可用于燃油消耗惩罚（action != 0 视为使用引擎）
- **其他衍生信号**：
  - 距离目标：`dist = sqrt(x^2 + y^2)`（可直接计算）
  - 速度大小：`speed = sqrt(vx^2 + vy^2)`
  - 距离减少：`delta_dist = dist_prev - dist_next`
  - 姿态偏离：`angle_deviation`（假设水平为0）

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
| 1 | orientation_penalty + progress_delta + soft_landing | 163.33 | 163.33 | 0.00 | 436.40 | orientation_penalty=-0.001 progress_delta=0.011 soft_landing=0.118 | new_best |
| 2 | orientation_penalty + progress_delta + soft_landing | -30.30 | 163.33 | -193.63 | 95.85 | orientation_penalty=-0.001 progress_delta=0.016 soft_landing=0.014 | no_meaningful_improvement |
| 3 | orientation_penalty + safe_progress + soft_landing | -11.06 | 163.33 | -174.39 | 1000.00 | orientation_penalty=-0.003 safe_progress=0.003 soft_landing=0.474 | no_meaningful_improvement |
| 4 | orientation_penalty + safe_progress + soft_landing | 146.42 | 163.33 | -16.91 | 812.95 | orientation_penalty=-0.000 safe_progress=0.005 soft_landing=0.373 | unsolved_high_achievement_continue_from_best |
| 5 | approach + orientation_penalty + soft_landing | -23.46 | 163.33 | -186.79 | 1000.00 | approach=0.596 orientation_penalty=-0.001 soft_landing=0.459 | no_meaningful_improvement |
| 6 | orientation_penalty + proximity_reward + soft_landing | 194.63 | 194.63 | 0.00 | 630.20 | orientation_penalty=-0.001 proximity_reward=0.750 soft_landing=0.375 | new_best |
| 7 | proximity_reward + soft_landing + stability_penalty | -10.97 | 194.63 | -205.60 | 1000.00 | proximity_reward=0.736 soft_landing=0.343 stability_penalty=-0.010 | no_meaningful_improvement |
| 8 | attitude_penalty + proximity_reward + soft_landing | -10.72 | 194.63 | -205.34 | 1000.00 | attitude_penalty=-0.000 proximity_reward=0.719 soft_landing=0.330 | no_meaningful_improvement |
