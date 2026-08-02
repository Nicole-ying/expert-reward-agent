# 1. Search objective
- target_score: 200.000000
- current_score: -115.677399
- gap_to_target: 315.677399
- target_achievement_ratio: -57.839%

# 2. 上一轮奖励函数代码（该轮得分: -115.677399）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重和阈值
    w_progress = 1.0
    w_angle = 0.5
    w_angvel = 0.1
    w_soft_land = 2.0
    w_eff = 0.02

    angle_thresh = 0.3   # rad
    angvel_thresh = 1.0  # rad/s
    max_speed_land = 1.0 # 着陆容许最大合速度
    max_angle_land = 0.5 # 着陆容许最大倾角 rad
    max_safe_vy = 0.5    # 安全下降的垂直速度阈值（m/s）

    # 距离进展（步间距离减少量），加入安全下降门控
    old_dist = (obs[0]**2 + obs[1]**2)**0.5
    new_dist = (next_obs[0]**2 + next_obs[1]**2)**0.5
    delta_dist = old_dist - new_dist   # 正值表示向目标接近

    vy = next_obs[3]                   # 垂直速度
    downward_speed = -vy if vy < 0.0 else 0.0   # 向下速度
    if downward_speed > max_safe_vy:
        overshoot = downward_speed - max_safe_vy
        gate = max(0.0, 1.0 - overshoot / max_safe_vy)
    else:
        gate = 1.0
    progress = w_progress * delta_dist * gate

    # 姿态稳定性（hinge 惩罚）
    angle = next_obs[4]
    angvel = next_obs[5]
    angle_penalty = -w_angle * max(0.0, abs(angle) - angle_thresh)
    angvel_penalty = -w_angvel * max(0.0, abs(angvel) - angvel_thresh)

    # 软着陆奖励（仅在支撑腿接触时有效）
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    any_contact = 1.0 if (left_contact + right_contact) > 0.5 else 0.0

    speed = (next_obs[2]**2 + next_obs[3]**2)**0.5
    speed_factor = 1.0 - min(1.0, speed / max_speed_land)
    angle_factor = 1.0 - min(1.0, abs(angle) / max_angle_land)
    soft_landing_score = speed_factor * angle_factor
    soft_landing = w_soft_land * soft_landing_score * any_contact

    # 发动机使用惩罚（离散动作每次非零动作）
    eff_penalty = -w_eff * (0.0 if action == 0 else 1.0)

    total_reward = progress + angle_penalty + angvel_penalty + soft_landing + eff_penalty

    components = {
        'progress': progress,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty,
        'soft_landing': soft_landing,
        'efficiency': eff_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.45 | -113.71 | ✅ |
| 2 | 骨架变化: angle_penalty + angvel_penalty + efficiency + prog | — | 68.35 | -115.68 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-115.677399, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.820934, -92.536457]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 0.777300 | 56.6% | 56.6% | 1.2% |
| angvel_penalty | -0.298458 | -21.7% | 21.7% | 1.0% |
| progress | 0.199347 | 14.5% | 16.7% | 53.5% |
| efficiency | -0.069000 | -5.0% | 5.0% | 5.0% |
| angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Mean eval score -115.68, all episodes crash early at ~68 steps. Shaped reward per step +0.0074 but true env reward -1.6156, indicating severe proxy misalignment.

**Component Anomalies**: soft_landing dominates (56.6% signed share) yet active only 1.2% of steps. angle_penalty dead (0 mean, 0% active). angvel_penalty rare (1% active) but significant negative share (-21.7%).

**Training Dynamics**: No checkpoint snapshots; only final policy data. Temporal trends unavailable.

**Signal Quality**: Dead angle_penalty threshold never crossed. Sparse soft_landing reward (1.2% active) likely causes high variance. Positive shaped reward fails to reflect crashing outcome.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 飞行器轨迹优化任务。初始时，飞行器从视口顶部中心附近以随机初速度开始运动。核心目标是**尽快且平稳地降落在中央目标垫上**，同时**尽量减少发动机使用**。学习者必须学会接近目标、主动减速、保持机身姿态稳定，并在所有支撑腿（或接触点）安全触垫后停止动作。失败的着陆（碰撞、飞出水平边界）应彻底避免。注意：节能是次目标，不应凌驾于成功着陆之上。

## 3. 观察空间 observation_space
- type: Box（连续向量）
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x_position —— 飞行器中心相对目标垫中心的水平坐标。reward_usable: true
- obs[1]: y_position —— 飞行器底部（或重心）相对垫面的垂直高度。reward_usable: true
- obs[2]: x_velocity —— 水平线速度。reward_usable: true
- obs[3]: y_velocity —— 垂直线速度。reward_usable: true
- obs[4]: body_angle —— 机身倾斜角。reward_usable: true
- obs[5]: angular_velocity —— 角速度。reward_usable: true
- obs[6]: left_support_contact —— 左支撑/腿触地标志（0/1）。reward_usable: true
- obs[7]: right_support_contact —— 右支撑/腿触地标志（0/1）。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine（无任何推力，仅靠惯性/重力）  
- action 1: left_orientation_engine（启动左侧姿态发动机，产生逆时针/向右的力矩）  
- action 2: main_engine（启动主发动机，向上产生推力）  
- action 3: right_orientation_engine（启动右侧姿态发动机，产生顺时针/向左的力矩）  

主发动机提供垂直方向推力，姿态发动机用于调整倾斜角度从而改变水平推力方向，典型的**主推力+姿态控制**模式。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  终止条件中的 `body_not_awake_or_settled` 极可能对应成功着陆后的稳定状态：飞行器停在目标垫上，速度降至零，物理引擎判定其静止休眠。
- failure-like termination:  
  - `crash_or_body_contact`：飞行器主体（非支撑腿）触地或与障碍物碰撞。  
  - `horizontal_position_outside_viewport`：水平坐标超出视口范围。
- ambiguous termination:  
  理论上 `body_not_awake_or_settled` 也可能因其它原因（如卡在边界外）触发，但实际环境中通常与成功着陆强关联。
- truncation: 本环境无时间截断（step 返回 truncated 固定为 False）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空字典）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 无任何 info 字段
- 成功/失败只能**间接推断**：  
  当 episode 终止时，通过最终观测（位置、速度、角度、接触标志）综合判断是否成功着陆：
  - 推断成功条件：`|x_position| < ε_x`，`|y_position| < ε_y`，`√(vx²+vy²) < ε_vel`，`|body_angle| < ε_ang`，且至少一个接触标志为 1。
  - 否则为失败。  
  该推断路径标记为 **derived_possible**。

## 7. 可用于奖励函数的信号
**位置与接近度**：
- `next_obs[0]`（x 偏移）：越靠近 0 越好。
- `next_obs[1]`（y 偏移）：越靠近 0 越好，且应在安全范围内。
- 可直接计算距离 `dist = sqrt(x² + y²)` 或步间距离减少量 `delta_dist`。

**速度**：
- `next_obs[2]`（vx）、`next_obs[3]`（vy）：着陆时需极低速度，飞行中可适度引导向下减速。

**姿态**：
- `next_obs[4]`（body_angle）：应保持在安全区间内（例如 ±0.3 rad），防止侧翻。
- `next_obs[5]`（angular_velocity）：应趋近 0。

**接触**：
- `next_obs[6]`（左触地）、`next_obs[7]`（右触地）：成功着陆通常需要双接触（或至少一腿触地且速度达标）。

**动作**：
- `action` 可用来惩罚发动机使用（尤其 punishment for engine actions），但禁止奖励“无动作”。

**终端事件（derived_possible）**：
- 由最终 `next_obs` 推断的成功奖励（大正分）或失败惩罚（负分）。

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
| 1 | angle_penalty + angvel_penalty + efficiency + progress + soft_landing | -113.71 | -113.71 | 0.00 | 68.45 | angle_penalty=-0.001 angvel_penalty=-0.003 efficiency=-0.003 progress=0.016 soft_landing=0.012 | new_best |
| 2 | angle_penalty + angvel_penalty + efficiency + progress + soft_landing | -115.68 | -113.71 | -1.96 | 68.35 | angle_penalty=-0.001 angvel_penalty=-0.003 efficiency=-0.003 progress=0.003 soft_landing=0.012 | no_meaningful_improvement |
