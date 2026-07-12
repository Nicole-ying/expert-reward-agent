# 1. Search objective
- target_score: 300.000000
- current_score: -65.961915
- gap_to_target: 365.961915
- target_achievement_ratio: -21.987%

# 2. 上一轮奖励函数代码（该轮得分: -65.961915）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号
    #    使用 horizontal_velocity (obs[2]) 作为前进激励
    #    采用凸化形式 (signal**2) 打破低速平台，鼓励更快行走
    # ============================================================
    horizontal_vel = obs[2]
    forward_progress_reward = 1.0 * (horizontal_vel ** 2)

    # ============================================================
    # 2. balance_maintenance: 稳定/安全约束
    #    使用 hull_angle (obs[0]) 和 hull_angular_velocity (obs[1])
    #    采用 quadratic_penalty 形式，对倾斜角度和角速度施加轻量惩罚
    #    权重较小，避免压制探索
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    balance_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_vel ** 2)

    # ============================================================
    # 3. gait_cycle_encouragement: 步态周期鼓励
    #    使用 leg1_contact (obs[8]) 和 leg2_contact (obs[13])
    #    鼓励双腿交替接触地面，惩罚双腿同时离地
    #    采用 binary_state_signal 组合形式
    # ============================================================
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    # 鼓励至少一条腿接触地面（避免跳跃/蹦跳）
    contact_sum = leg1_contact + leg2_contact
    # 当双腿同时离地时给予惩罚
    both_off_ground_penalty = -0.2 * (1.0 - contact_sum) if contact_sum < 0.5 else 0.0
    # 鼓励交替接触（当一条腿接触时给予小奖励）
    alternating_bonus = 0.1 * (leg1_contact * (1.0 - leg2_contact) + leg2_contact * (1.0 - leg1_contact))

    # ============================================================
    # 组合总奖励
    # ============================================================
    total_reward = forward_progress_reward + balance_penalty + both_off_ground_penalty + alternating_bonus

    # ============================================================
    # 组件记录
    # ============================================================
    components = {
        "forward_progress_reward": forward_progress_reward,
        "balance_penalty": balance_penalty,
        "both_off_ground_penalty": both_off_ground_penalty,
        "alternating_bonus": alternating_bonus
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=-65.961915, len=1600.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-99.664145, -52.310843]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| alternating_bonus | 144.335000 | 79.1% | 79.1% | 90.2% |
| both_off_ground_penalty | -21.400000 | -11.7% | 11.7% | 6.7% |
| balance_penalty | -9.211715 | -5.0% | 5.0% | 100.0% |
| forward_progress_reward | 7.553309 | 4.1% | 4.1% | 84.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务的核心目标是控制一个双足机器人，在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。机器人需要协调两条腿的髋关节和膝关节，产生稳定的双足步态。摔倒会导致回合终止。附属目标（省能量、速度快）是优化方向，但主目标是“持续前进通过地形”。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 (推测)
- obs[0]: hull_angle，主体相对于竖直方向的角度，reward_usable: true
- obs[1]: hull_angular_velocity，主体角速度，reward_usable: true
- obs[2]: horizontal_velocity，前后线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿1髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿1地面接触标志(1.0=接触, 0.0=无接触)，reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2地面接触标志(1.0=接触, 0.0=无接触)，reward_usable: true
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
- success-like termination: reached_end_of_terrain（到达地形终点，可视为成功完成）
- failure-like termination: body_fallen_over（摔倒，明确失败）
- ambiguous termination: 无
- truncation: 无（truncated 始终为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空 {}，无 success 标志）
- explicit_failure_flag_available: false（info 字典为空 {}，无 failure 标志）
- allowed_info_fields: 无（info 始终为 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，可通过 horizontal_velocity 积分或 LIDAR 间接推断
- velocity: horizontal_velocity (obs[2])，vertical_velocity (obs[3])
- orientation: hull_angle (obs[0])，hull_angular_velocity (obs[1])
- contact: leg1_contact (obs[8])，leg2_contact (obs[13])
- action/engine: action 向量（4维扭矩），可用于能耗惩罚
- other: LIDAR 距离测量 (obs[14..23])，关节角度和速度 (obs[4..7], obs[9..12])

# 6. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 7. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | alternating_bonus + balance_penalty + both_off_ground_penalty + forward_progress_reward | -65.96 | -65.96 | 0.00 | 1600.00 | alternating_bonus=0.064 balance_penalty=-0.008 both_off_ground_penalty=-0.024 forward_progress_reward=0.004 | new_best |
