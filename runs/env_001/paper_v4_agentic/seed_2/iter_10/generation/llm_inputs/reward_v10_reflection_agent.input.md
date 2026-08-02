# 1. Search objective
- target_score: 200.000000
- current_score: 145.891679
- gap_to_target: 54.108321
- target_achievement_ratio: 72.946%

# 2. 上一轮奖励函数代码（该轮得分: 145.891679）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- Extract next_obs signals ---
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Helper: distance from pad center ---
    horizontal_dist = abs(x)
    distance_to_target = (horizontal_dist**2 + y**2) ** 0.5

    # --- Component A: main progress signal via potential-based shaping ---
    norm_distance = distance_to_target / 2.5
    angle_penalty = abs(angle) / 1.57
    potential = -(norm_distance + 0.3 * angle_penalty)

    prev_x = obs[0]
    prev_y = obs[1]
    prev_angle = obs[4]
    prev_horizontal_dist = abs(prev_x)
    prev_distance = (prev_horizontal_dist**2 + prev_y**2) ** 0.5
    prev_norm_distance = prev_distance / 2.5
    prev_angle_penalty = abs(prev_angle) / 1.57
    prev_potential = -(prev_norm_distance + 0.3 * prev_angle_penalty)

    potential_delta = potential - prev_potential

    # --- Component B: soft velocity health gate (unchanged) ---
    speed = (vx**2 + vy**2) ** 0.5
    safe_speed = 0.3 + 1.5 * distance_to_target
    overspeed_ratio = speed / (safe_speed + 1e-6)
    gate = 0.3 + 0.7 * (2.718281828 ** (-max(0.0, overspeed_ratio - 1.0)))

    scaled_progress = potential_delta * 10.0
    gated_progress = scaled_progress * gate

    # --- Component C: landing steady-state reward (TIGHT thresholds, revert iter8 relaxations) ---
    # Distance factor: linear decay, zero beyond 0.2 (tightened from 0.25)
    dist_factor = max(0.0, 1.0 - distance_to_target / 0.2)

    # Contact: both legs
    contact_factor = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0

    # Speed factor: full score below 0.15, linear decay to 0 at 0.35 (tightened from 0.3/0.5)
    if speed < 0.15:
        speed_factor = 1.0
    else:
        speed_factor = max(0.0, 1.0 - (speed - 0.15) / 0.2)

    # Angle factor: full score below 0.15, decay to 0 at 0.35 (tightened from 0.25/0.45)
    if abs(angle) < 0.15:
        angle_factor = 1.0
    else:
        angle_factor = max(0.0, 1.0 - (abs(angle) - 0.15) / 0.2)

    # Angular velocity factor: full score below 0.5, decay to 0 at 1.0 (tightened from 0.8/1.2)
    if abs(angular_vel) < 0.5:
        angular_factor = 1.0
    else:
        angular_factor = max(0.0, 1.0 - (abs(angular_vel) - 0.5) / 0.5)

    landing_factor = dist_factor * contact_factor * speed_factor * angle_factor * angular_factor
    C_landing = 0.15 * landing_factor   # coefficient unchanged

    total_reward = gated_progress + C_landing

    components = {
        'A_progress_gated': gated_progress,
        'C_landing_steady': C_landing
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 134.15 | -116.46 | ✅ |
| 2 | velocity_danger将在整个下降过程中提供梯度递减的减速压力，proximity_delta放大50×后... | velocity_danger将在整个下降过程中提供梯度递减的减速压力，proximity_delta放大50×后... | 68.45 | -110.22 | ✅ |
| 3 | 引入 soft_landing_bonus 后，agent 会尝试在接近目标时减速、对齐姿态并触发腿接触，从而获得... | 引入 soft_landing_bonus 后，agent 会尝试在接近目标时减速、对齐姿态并触发腿接触，从而获得... | 68.40 | -114.97 | ❌ |
| 4 | 放宽 `landing_bonus` 阈值将使其在着陆瞬间被激活，提供可学习的软着陆梯度，改善终端速度/姿态，延长... | 放宽 `landing_bonus` 阈值将使其在着陆瞬间被激活，提供可学习的软着陆梯度，改善终端速度/姿态，延长... | 68.40 | -111.88 | ❌ |
| 5 | 新组件在低高度时奖励低速与正姿态，提供可微分梯度引导 agent 在接近着陆垫时减速并调姿，从而延长 episod... | 新组件在低高度时奖励低速与正姿态，提供可微分梯度引导 agent 在接近着陆垫时减速并调姿，从而延长 episod... | 68.40 | -115.17 | ❓ |
| 6 | 骨架变化: A_progress_gated + C_landing_steady | — | 917.45 | 135.20 | ✅ |
| 7 | 放宽 speed_gate 允许合理下降速度通过，并放大 progress 尺度，将恢复从起点到目标的有效引导梯度... | 放宽 speed_gate 允许合理下降速度通过，并放大 progress 尺度，将恢复从起点到目标的有效引导梯度... | 1000.00 | 145.47 | ✅ |
| 8 | 放宽 speed_factor 阈值（0.1→0.3/0.5）和 angle_factor 阈值（0.1→0.25... | 放宽 speed_factor 阈值（0.1→0.3/0.5）和 angle_factor 阈值（0.1→0.25... | 955.20 | 141.79 | ❌ |
| 9 | 收紧着陆条件阈值（速度/角度/角速度/距离门槛）将迫使 agent 追求更精确的着陆姿态，从而降低 exploit... | 收紧着陆条件阈值（速度/角度/角速度/距离门槛）将迫使 agent 追求更精确的着陆姿态，从而降低 exploit... | 1000.00 | 145.89 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=145.891679, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[117.026921, 177.956643]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| C_landing_steady | 99.434808 | 93.7% | 93.7% | 71.8% |
| A_progress_gated | 5.568592 | 5.2% | 6.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Automatic fallback after 5 turns without submit. Raw data: [inspect_component_dynamics]: (no monitor snapshots — training may not have completed)
[inspect_training_feedback]: # Training Feedback

## Final-policy outcome
score=145.891679, len=1000.000000, term

**Component Anomalies**: Subagent exhausted turns without explicit submission.

**Training Dynamics**: No temporal analysis available.

**Signal Quality**: No signal quality assessment available.

**Evidence Confidence**: `low`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
任务主目标是控制一台具有左右支撑腿和主/侧向引擎的 2D 着陆器，从视野顶部中央附近以随机初始推力开始，尽快到达场地中央的着陆垫，并稳定、安全地停靠在该垫上。次要目标是尽量减少引擎使用（燃料消耗），同时保持车身姿态稳定，实现软着陆。不应将单纯的快速到达或单纯省燃料作为独立核心目标，代理必须在安全着陆的前提下兼顾速度与能效。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（根据环境惯例，具体从环境读取，但推测为 float）  
- obs[0]: x_position，相对目标垫的水平坐标，reward_usable: true  
- obs[1]: y_position，相对着陆垫高度的垂直坐标，reward_usable: true  
- obs[2]: x_velocity，水平线速度，reward_usable: true  
- obs[3]: y_velocity，垂直线速度，reward_usable: true  
- obs[4]: body_angle，车身俯仰/横滚角度，reward_usable: true  
- obs[5]: angular_velocity，角速度，reward_usable: true  
- obs[6]: left_support_contact，左支撑腿接触标志（0.0 或 1.0），reward_usable: true（需谨慎使用）  
- obs[7]: right_support_contact，右支撑腿接触标志（0.0 或 1.0），reward_usable: true（需谨慎使用）

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine，不点火，不做任何事情  
- action 1: left_orientation_engine，点燃左姿态引擎（产生方向性的力）  
- action 2: main_engine，点燃主引擎（通常向上推力）  
- action 3: right_orientation_engine，点燃右姿态引擎（与左引擎反向）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 当 `body_not_awake_or_settled` 为真，且未发生 crash/出界时，可推断为成功着陆并稳定。
- failure-like termination: `crash_or_body_contact`（猛烈碰撞或非法身体接触）和 `horizontal_position_outside_viewport`（水平飞出视野）均视为失败。
- ambiguous termination: 单独的 `body_not_awake_or_settled` 可能发生在成功着陆后不久，也可能是因摔落后静止，需结合位置、速度判断。
- truncation: 环境中未提供 truncation 信号（step 返回 `terminated, False`），因此不存在时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（step 返回 info={}）  
- forbidden_or_uncertain_info_fields: 任何 info 字典中的字段均不存在，不可使用

间接推断路径：
- 成功着陆可从 termination 时接近目标垫中心、垂直速度极低、角度接近 0 且至少一腿接触的条件组合推断（derived_possible）。
- crash 可从 termination 时 y_position 骤降至地面以下或水平位置远超边界等条件间接检测（derived_possible）。

## 7. 可用于奖励函数的信号
- position：
  - 相对垫的水平距离 `|x_position|`
  - 相对垫高度的垂直距离 `|y_position|`（当 y_position 为正代表高于垫，到达垫面时理想 y≈0）
- velocity：
  - 水平速度 `x_velocity`（软着陆要求接近0）
  - 垂直速度 `y_velocity`（负值代表下落，着陆瞬间需要小）
- orientation：
  - body_angle（理想接近0，可取其绝对值或二次惩罚）
  - angular_velocity（软着陆应接近0）
- contact：
  - left_support_contact / right_support_contact（至少一腿接触可能表明着陆成功，但需结合速度和位置，否则可能鼓励猛烈砸地）
- action/engine：
  - action 本身（可计算引擎使用惩罚，no_engine 时无推力）
- other：
  - 通过 next_obs 与 obs 的差值得出速度变化，可用于检测剧烈推力
  - 衍生信号：是否接近目标垫且速度降低（derived_possible）

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
| 1 | orientation_penalty + proximity_delta + velocity_penalty | -116.46 | -116.46 | 0.00 | 134.15 | orientation_penalty=-0.003 proximity_delta=0.005 velocity_penalty=-0.002 | new_best |
| 2 | orientation_penalty + proximity_delta + velocity_danger | -110.22 | -110.22 | 0.00 | 68.45 | orientation_penalty=-0.081 proximity_delta=0.776 velocity_danger=-0.117 | new_best |
| 3 | landing_bonus + orientation_penalty + proximity_delta + velocity_danger | -114.97 | -110.22 | -4.74 | 68.40 | landing_bonus=0.044 orientation_penalty=-0.078 proximity_delta=0.781 velocity_danger=-0.118 | no_meaningful_improvement |
| 4 | landing_bonus + orientation_penalty + proximity_delta + velocity_danger | -111.88 | -110.22 | -1.66 | 68.40 | landing_bonus=0.010 orientation_penalty=-0.112 proximity_delta=0.787 velocity_danger=-0.120 | no_meaningful_improvement |
| 5 | orientation_penalty + proximity_delta + soft_approach_bonus + velocity_danger | -115.17 | -110.22 | -4.94 | 68.40 | orientation_penalty=-0.074 proximity_delta=0.789 soft_approach_bonus=0.015 velocity_danger=-0.121 | unsolved_stagnation_fresh_restart |
| 6 | A_progress_gated + C_landing_steady | 135.20 | 135.20 | 0.00 | 917.45 | A_progress_gated=0.001 C_landing_steady=0.137 | new_best |
| 7 | A_progress_gated + C_landing_steady | 145.47 | 145.47 | 0.00 | 1000.00 | A_progress_gated=0.011 C_landing_steady=0.139 | new_best |
| 8 | A_progress_gated + C_landing_steady | 141.79 | 145.47 | -3.68 | 955.20 | A_progress_gated=0.010 C_landing_steady=0.087 | no_meaningful_improvement |
| 9 | A_progress_gated + C_landing_steady | 145.89 | 145.89 | 0.00 | 1000.00 | A_progress_gated=0.010 C_landing_steady=0.082 | new_best |
