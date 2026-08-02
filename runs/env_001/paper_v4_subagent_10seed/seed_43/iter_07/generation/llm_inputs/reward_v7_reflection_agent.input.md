# 1. Search objective
- target_score: 200.000000
- current_score: -117.778715
- gap_to_target: 317.778715
- target_achievement_ratio: -58.889%

# 2. 上一轮奖励函数代码（该轮得分: -117.778715）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    # left_c = obs[6], right_c = obs[7] not used for current state

    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # 1. Soft landing proxy reward (main learning signal)
    landing_reward = 0.0
    if next_left > 0.5 and next_right > 0.5:
        # Position factor: prefer x close to 0
        pos_factor = 2.718281828 ** (-(next_x ** 2) / (2 * 0.0025))  # sigma = 0.05
        # Speed factor: penalise high total speed
        speed_n = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
        spd_factor = 2.718281828 ** (-(speed_n ** 2) / (2 * 0.04))   # sigma = 0.2
        # Attitude factor: prefer upright
        ang_n = abs(next_angle)
        ang_factor = 2.718281828 ** (-(ang_n ** 2) / (2 * 0.01))     # sigma = 0.1
        landing_reward = 10.0 * pos_factor * spd_factor * ang_factor

    # 2. Progress reward: reduction in distance to target (auxiliary)
    dist_now = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    delta_dist = dist_now - dist_next

    # Safety gate for progress: when near target, suppress reward if speed/angle are high
    near_target = dist_now < 0.5
    gate = 1.0
    if near_target:
        # Use current vertical speed and body angle to form a soft gate
        gate = 1.0 / (1.0 + 10.0 * (y_vel ** 2) + 5.0 * (angle ** 2))
    progress_reward = delta_dist * gate

    # 3. Action efficiency penalty (very small)
    action_cost = -0.01 if action != 0 else 0.0

    # 4. Boundary penalty: discourage moving outside viewport horizontally
    boundary_penalty = 0.0
    if abs(x_pos) > 1.0:
        boundary_penalty = -5.0 * (abs(x_pos) - 1.0)

    # 5. Light angle/angular-velocity penalty (global, to stabilise attitude)
    angle_penalty = -0.01 * abs(angle) - 0.001 * abs(ang_vel)

    total_reward = (landing_reward + progress_reward +
                    action_cost + boundary_penalty + angle_penalty)

    components = {
        "landing_soft_reward": landing_reward,
        "progress": progress_reward,
        "action_cost": action_cost,
        "boundary_penalty": boundary_penalty,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 775.10 | -115.30 | ✅ |
| 2 | 新奖励在高度低、速度小、姿态好的区间提供密集梯度，引导 agent 进入着陆走廊；active_rate 将从 0... | 新奖励在高度低、速度小、姿态好的区间提供密集梯度，引导 agent 进入着陆走廊；active_rate 将从 0... | 1000.00 | -18.80 | ✅ |
| 3 | 用稀疏但高额的接触成功奖励替换 exploit 源，使 agent 只有真正安全双腿着陆才能获得最大回报，prog... | 用稀疏但高额的接触成功奖励替换 exploit 源，使 agent 只有真正安全双腿着陆才能获得最大回报，prog... | 501.05 | -112.84 | ❌ |
| 4 | 移除惩罚并将其转化为 progress 的门控因子，消除“越快死越赚”的激励，agent 将恢复生存并逐步改善姿态... | 移除惩罚并将其转化为 progress 的门控因子，消除“越快死越赚”的激励，agent 将恢复生存并逐步改善姿态... | 131.15 | -115.49 | ❌ |
| 5 | 恢复密集的接近‑安全奖励（landing_approach_reward）将为 agent 提供连续梯度，引导其靠... | 恢复密集的接近‑安全奖励（landing_approach_reward）将为 agent 提供连续梯度，引导其靠... | 1000.00 | -55.78 | ❓ |
| 6 | 骨架变化: action_cost + angle_penalty + boundary_penalty + l | — | 68.30 | -117.78 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-117.778715, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.853315, -98.092066]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 0.923771 | 64.2% | 66.0% | 100.0% |
| landing_soft_reward | 0.413275 | 28.7% | 28.7% | 0.7% |
| angle_penalty | -0.048717 | -3.4% | 3.4% | 100.0% |
| action_cost | -0.027500 | -1.9% | 1.9% | 4.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Env reward -117.8 (all episodes early-term <150 steps, score<-50) while shaped reward per ep sums to ~+1.26. Shaped reward mismatch: progress (64% share) and sparse landing reward (29% share, active 0.7% steps) dominate, but env outcome catastrophic.

**Component Anomalies**: boundary_penalty dead (zero always). landing_soft_reward active only 0.7% steps but 29% magnitude share. progress always active (100%) dominates 64% signed share. action_cost near-dead (4% active). angle_penalty always on but -3.4% share.

**Training Dynamics**: No checkpoint snapshots provided; unable to assess temporal trends.

**Signal Quality**: Dead gates: boundary_penalty never fires. Missing dense landing guide: landing reward triggers rarely. Progress gate may suppress near-target. No component signals steady approach to soft landing.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
该任务是一个 2D 飞行器（或着陆器）的精确软着陆问题。一个带两条支撑腿的飞行器从上方某个位置开始，施加一个随机初始力。核心目标是在最短时间内以最少的发动机推力安全降落在中心目标平台上，实现两条支撑腿同时接触平台、姿态接近垂直、速度几乎为零的稳定停靠。Agent 必须学会高效地靠近目标区域、减速、保持姿态稳定并建立安全接触。次要目标是降低发动机使用频率和总动作数，以节省燃料。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float (likely float32)
- obs[0] (`x_position`): 水平坐标，相对于目标平台的中心。可用于计算到目标的水平距离。
  - reward_usable: true
- obs[1] (`y_position`): 垂直坐标，相对于目标平台的高度。可用于高度/距离计算。
  - reward_usable: true
- obs[2] (`x_velocity`): 水平线速度。用于速度惩罚或接触条件。
  - reward_usable: true
- obs[3] (`y_velocity`): 垂直线速度。用于着陆软硬判定。
  - reward_usable: true
- obs[4] (`body_angle`): 身体朝向角度（以弧度计，0 表示竖直）。用于姿态稳定性约束。
  - reward_usable: true
- obs[5] (`angular_velocity`): 角速度。用于姿态变化惩罚。
  - reward_usable: true
- obs[6] (`left_support_contact`): 左支撑腿接触目标平台标志（1.0 接触，0.0 未接触）。关键着陆信号。
  - reward_usable: true
- obs[7] (`right_support_contact`): 右支撑腿接触目标平台标志。关键着陆信号。
  - reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` – 不启动任何发动机，滑行。
- action 1: `left_orientation_engine` – 启动左侧姿态发动机，产生向左旋转的力矩，调整身体角度。
- action 2: `main_engine` – 启动主发动机，提供向上的推力（对抗重力或减速）。
- action 3: `right_orientation_engine` – 启动右侧姿态发动机，产生向右旋转的力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 可能由 `body_not_awake_or_settled` 触发，当飞行器两条腿都接触目标平台、速度极低、姿态稳定时，身体被判定为“settled”，随后 episode 终止。虽然环境没有提供显式成功标志，但这一条件可作为成功完成的代理。
- failure-like termination: `crash_or_body_contact`（身体其他部位撞击地面或平台）和 `horizontal_position_outside_viewport`（水平飞出视野）都是明显的失败终止。
- ambiguous termination: `body_not_awake_or_settled` 也可能在不稳定或仅单腿接触的情况下触发，因而单独不代表成功，需要结合其他观测区分。
- truncation: 无截断（环境未设定最大步数

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
| 1 | landing_bonus + progress + soft_landing_penalty | -115.30 | -115.30 | 0.00 | 775.10 | landing_bonus=7.699 progress=0.040 soft_landing_penalty=0.203 | new_best |
| 2 | landing_approach_reward + progress + soft_landing_penalty | -18.80 | -18.80 | 0.00 | 1000.00 | landing_approach_reward=2.424 progress=0.036 soft_landing_penalty=0.198 | new_best |
| 3 | contact_success_reward + progress + soft_landing_penalty | -112.84 | -18.80 | -94.04 | 501.05 | contact_success_reward=42.715 progress=0.051 soft_landing_penalty=0.281 | no_meaningful_improvement |
| 4 | contact_success_reward + landing_gate + progress | -115.49 | -18.80 | -96.69 | 131.15 | contact_success_reward=0.722 landing_gate=0.145 progress=0.117 | no_meaningful_improvement |
| 5 | contact_success_reward + landing_approach_reward + progress | -55.78 | -18.80 | -36.97 | 1000.00 | contact_success_reward=50.158 landing_approach_reward=0.424 progress=0.029 | unsolved_stagnation_fresh_restart |
| 6 | action_cost + angle_penalty + boundary_penalty + landing_soft_reward + progress | -117.78 | -18.80 | -98.98 | 68.30 | action_cost=-0.003 angle_penalty=-0.001 boundary_penalty=0.000 landing_soft_reward=0.013 progress=0.013 | no_meaningful_improvement |
