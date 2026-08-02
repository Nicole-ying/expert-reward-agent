# Duplicate reward retry
The previous generation duplicated iter 2 (runs\env_001\paper_v4_subagent_10seed\seed_43\iter_02\generation\reward_v2.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward (improvement_delta on distance to origin)
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = 10.0 * (dist_curr - dist_next)  # positive when getting closer

    # 2. Soft landing constraints
    k = 0.5
    desired_y_vel = -k * y_next
    vertical_error = y_vel_next - desired_y_vel
    penalty_y_vel = vertical_error**2
    penalty_x_vel = x_vel_next**2
    angle_error = max(0.0, abs(angle_next) - 0.2)
    penalty_angle = angle_error**2
    ang_vel_error = max(0.0, abs(ang_vel_next) - 0.5)
    penalty_ang_vel = ang_vel_error**2

    w_y_vel = 1.0
    w_x_vel = 1.0
    w_angle = 2.0
    w_ang_vel = 0.5

    soft_landing_penalty = (w_y_vel * penalty_y_vel +
                            w_x_vel * penalty_x_vel +
                            w_angle * penalty_angle +
                            w_ang_vel * penalty_ang_vel)

    # 3. Landing approach reward (continuous bounded factor)
    h_eff = y_next if y_next > 0.0 else 0.0

    v_desired = -0.5 * h_eff

    sigma_h = 0.5
    sigma_vx = 0.2
    sigma_vy = 0.3
    sigma_angle = 0.15

    factor_height = 2.718281828 ** (- (h_eff**2) / (sigma_h**2))
    factor_vx = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx**2))
    factor_vy = 2.718281828 ** (- ((y_vel_next - v_desired)**2) / (sigma_vy**2))
    factor_angle = 2.718281828 ** (- (angle_next**2) / (sigma_angle**2))

    w_approach = 5.0
    landing_approach_reward = w_approach * factor_height * factor_vx * factor_vy * factor_angle

    total_reward = progress - soft_landing_penalty + landing_approach_reward

    components = {
        'progress': progress,
        'soft_landing_penalty': soft_landing_penalty,
        'landing_approach_reward': landing_approach_reward
    }

    return float(total_reward), components
```

# 1. Search objective
- target_score: 200.000000
- current_score: -18.800932
- gap_to_target: 218.800932
- target_achievement_ratio: -9.400%

# 2. 上一轮奖励函数代码（该轮得分: -18.800932）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward (improvement_delta on distance to origin)
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = 10.0 * (dist_curr - dist_next)  # positive when getting closer

    # 2. Soft landing constraints (unchanged from original, but known to be problematic)
    k = 0.5
    desired_y_vel = -k * y_next
    vertical_error = y_vel_next - desired_y_vel
    penalty_y_vel = vertical_error**2
    penalty_x_vel = x_vel_next**2
    angle_error = max(0.0, abs(angle_next) - 0.2)
    penalty_angle = angle_error**2
    ang_vel_error = max(0.0, abs(ang_vel_next) - 0.5)
    penalty_ang_vel = ang_vel_error**2

    w_y_vel = 1.0
    w_x_vel = 1.0
    w_angle = 2.0
    w_ang_vel = 0.5

    soft_landing_penalty = (w_y_vel * penalty_y_vel +
                            w_x_vel * penalty_x_vel +
                            w_angle * penalty_angle +
                            w_ang_vel * penalty_ang_vel)

    # 3. Landing approach reward (continuous bounded factor — REPLACES dead landing_bonus)
    # Effective height above ground (non-negative)
    h_eff = y_next if y_next > 0.0 else 0.0

    # Desired vertical speed for current height
    v_desired = -0.5 * h_eff

    # Gaussian factors: each in (0, 1], peaking at ideal condition
    sigma_h = 0.5
    sigma_vx = 0.2
    sigma_vy = 0.3
    sigma_angle = 0.15

    factor_height = 2.718281828 ** (- (h_eff**2) / (sigma_h**2))
    factor_vx = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx**2))
    factor_vy = 2.718281828 ** (- ((y_vel_next - v_desired)**2) / (sigma_vy**2))
    factor_angle = 2.718281828 ** (- (angle_next**2) / (sigma_angle**2))

    # Combined approach reward: high only when all conditions are near-ideal
    w_approach = 5.0
    landing_approach_reward = w_approach * factor_height * factor_vx * factor_vy * factor_angle

    total_reward = progress - soft_landing_penalty + landing_approach_reward

    components = {
        'progress': progress,
        'soft_landing_penalty': soft_landing_penalty,
        'landing_approach_reward': landing_approach_reward
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 775.10 | -115.30 | ✅ |
| 2 | 新奖励在高度低、速度小、姿态好的区间提供密集梯度，引导 agent 进入着陆走廊；active_rate 将从 0... | 新奖励在高度低、速度小、姿态好的区间提供密集梯度，引导 agent 进入着陆走廊；active_rate 将从 0... | 1000.00 | -18.80 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-18.800932, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-50.748072, 26.495458]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_approach_reward | 3138.190033 | 98.8% | 98.8% | 100.0% |
| soft_landing_penalty | 25.173090 | 0.8% | 0.8% | 100.0% |
| progress | 11.750811 | 0.4% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


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
