# Search objective
- target_score: 300.000000
- current_score: 61.768778
- gap_to_target: 238.231222
- target_achievement_ratio: 20.590%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 61.768778）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 信号提取 ==========
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    hip1_torque = action[0]
    knee1_torque = action[1]
    hip2_torque = action[2]
    knee2_torque = action[3]

    # ========== 组件 1: 前进奖励 (凸化) ==========
    # 将线性奖励改为平方形式，鼓励更高速度
    forward_reward = 3.0 * (horizontal_velocity ** 2)

    # ========== 组件 2: 平衡约束 (soft_health_gate) ==========
    angle_factor = 1.0 / (1.0 + 2.0 * abs(hull_angle))
    angular_velocity_factor = 1.0 / (1.0 + 0.5 * abs(hull_angular_velocity))
    balance_gate = min(angle_factor, angular_velocity_factor)
    gated_forward_reward = forward_reward * balance_gate

    # ========== 组件 3: 能耗惩罚 (quadratic_penalty) ==========
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (
        hip1_torque**2 + knee1_torque**2 + hip2_torque**2 + knee2_torque**2
    )

    # ========== 组件 4: 额外平衡惩罚 (hinge，保留但不常触发) ==========
    angle_safety_threshold = 0.8
    angle_penalty_weight = 1.0
    angle_penalty = -angle_penalty_weight * max(0.0, abs(hull_angle) - angle_safety_threshold)

    # ========== 总奖励 ==========
    total_reward = gated_forward_reward + energy_penalty + angle_penalty

    components = {
        "gated_forward_reward": gated_forward_reward,
        "energy_penalty": energy_penalty,
        "angle_penalty": angle_penalty,
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=61.768778, len=438.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-36.415345, 321.058772]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward_reward | 392.684694 | 98.7% | 98.7% | 100.0% |
| energy_penalty | -5.291401 | -1.3% | 1.3% | 100.0% |
| angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标是控制一个双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。次目标是保持身体平衡不摔倒。不应混淆的目标是：任务不是到达某个特定目标点，也不是单纯保持平衡，而是持续前进与能耗之间的权衡。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 (推测)
- obs[0]: hull_angle，身体相对于竖直方向的角度，reward_usable: true
- obs[1]: hull_angular_velocity，身体角速度，reward_usable: true
- obs[2]: horizontal_velocity，前后线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿1髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿1地面接触标志(1.0=接触,0.0=无接触)，reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2地面接触标志(1.0=接触,0.0=无接触)，reward_usable: true
- obs[14..23]: lidar_0..9，10个LIDAR测距仪沿前方地形的距离测量值，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- continuous: true
- bounds: [-1.0, 1.0] per joint
- action[0]: hip_torque_leg1，腿1髋关节施加的扭矩
- action[1]: knee_torque_leg1，腿1膝关节施加的扭矩
- action[2]: hip_torque_leg2，腿2髋关节施加的扭矩
- action[3]: knee_torque_leg2，腿2膝关节施加的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（成功到达地形终点）
- failure-like termination: body_fallen_over（摔倒，任务失败）
- ambiguous termination: 无
- truncation: 无（truncated=False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（terminated=True 但无法区分是成功还是失败）
- explicit_failure_flag_available: false（同上）
- allowed_info_fields: 无（info 字典为空 {}）
- forbidden_or_uncertain_info_fields: 无（但无法从 terminated 中区分成功/失败）

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，但可通过 LIDAR 间接推断前进距离
- velocity: horizontal_velocity (obs[2]), vertical_velocity (obs[3])
- orientation: hull_angle (obs[0]), hull_angular_velocity (obs[1])
- contact: leg1_contact (obs[8]), leg2_contact (obs[13])
- action/engine: action[0..3]（扭矩值，可用于能耗惩罚）
- other: 关节角度/速度 (obs[4..7], obs[9..12])，LIDAR (obs[14..23])

# Compact expert route context
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + energy_penalty + gated_forward_reward | 264.12 | 264.12 | 0.00 | 746.05 | angle_penalty=-0.002 energy_penalty=-0.022 gated_forward_reward=0.558 | new_best |
| 2 | angle_penalty + energy_penalty + gated_forward_reward | 61.77 | 264.12 | -202.35 | 438.80 | angle_penalty=-0.001 energy_penalty=-0.017 gated_forward_reward=0.330 | no_meaningful_improvement |
