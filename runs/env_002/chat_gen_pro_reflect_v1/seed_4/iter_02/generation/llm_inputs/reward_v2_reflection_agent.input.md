# 1. Search objective
- target_score: 300.000000
- current_score: 166.539950
- gap_to_target: 133.460050
- target_achievement_ratio: 55.513%

# 2. 上一轮奖励函数代码（该轮得分: 166.539950）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    # forward_progress: horizontal_velocity (obs[2])
    horizontal_vel = obs[2]
    
    # balance_maintenance: hull_angle (obs[0]), hull_angular_velocity (obs[1])
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    
    # ============================================================
    # Component 1: forward_progress (main learning signal)
    # Use dense_state_signal: positive linear reward for forward velocity
    # Scale: typical horizontal velocity range ~0-5 m/s, target ~2-3 m/s
    # Weight chosen so that at 2 m/s, this component ≈ 2.0
    # ============================================================
    forward_reward = 1.0 * horizontal_vel
    
    # ============================================================
    # Component 2: balance_maintenance (stability constraint)
    # Use dense_state_signal with quadratic penalty for hull_angle deviation
    # hull_angle is in radians, typical stable range ~[-0.3, 0.3]
    # Penalty is mild: at 0.3 rad (~17 deg), penalty ≈ -0.09
    # Also add small penalty for angular velocity to discourage wobbling
    # ============================================================
    angle_penalty = -2.0 * (hull_angle ** 2)
    angular_vel_penalty = -0.1 * (hull_angular_vel ** 2)
    balance_penalty = angle_penalty + angular_vel_penalty
    
    # ============================================================
    # Component 3: soft_health_gate on forward_reward
    # When hull_angle approaches dangerous territory (>0.5 rad), 
    # reduce forward_reward to discourage "rush then fall" behavior
    # Use linear decay gate: gate = max(0, 1 - |hull_angle| / 0.6)
    # At hull_angle=0.3, gate ≈ 0.5; at hull_angle=0.6, gate ≈ 0.0
    # ============================================================
    gate_factor = max(0.0, 1.0 - abs(hull_angle) / 0.6)
    gated_forward = forward_reward * gate_factor
    
    # ============================================================
    # Total reward
    # ============================================================
    total_reward = gated_forward + balance_penalty
    
    # ============================================================
    # Components dict (for debugging and monitoring)
    # ============================================================
    components = {
        'forward_reward': forward_reward,
        'gated_forward': gated_forward,
        'gate_factor': gate_factor,
        'angle_penalty': angle_penalty,
        'angular_vel_penalty': angular_vel_penalty,
        'balance_penalty': balance_penalty
    }
    
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=166.539950, len=820.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-43.184163, 295.075255]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 791.718914 | 51.3% | 51.3% | 100.0% |
| forward_reward | 381.440275 | 24.7% | 24.7% | 100.0% |
| gated_forward | 367.604345 | 23.8% | 23.8% | 100.0% |
| balance_penalty | -1.686407 | -0.1% | 0.1% | 100.0% |
| angle_penalty | -1.655801 | -0.1% | 0.1% | 100.0% |
| angular_vel_penalty | -0.030605 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标是控制一个双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。次目标是保持身体平衡不摔倒。不应混淆的目标是：虽然能量消耗是优化项，但核心驱动力是前进速度和距离，能量是附属约束；任务不是单纯保持平衡（生存），也不是精确导航到某个点，而是持续前进的 locomotion 任务。

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
- obs[8]: leg1_contact，腿1地面接触标志（1.0=接触，0.0=无接触），reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2地面接触标志（1.0=接触，0.0=无接触），reward_usable: true
- obs[14..23]: lidar_0..9，10个LIDAR测距仪沿前方地形的距离测量值，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- dtype: float32
- bounds: [-1.0, 1.0] per dimension
- action[0]: hip_torque_leg1，腿1髋关节施加的力矩
- action[1]: knee_torque_leg1，腿1膝关节施加的力矩
- action[2]: hip_torque_leg2，腿2髋关节施加的力矩
- action[3]: knee_torque_leg2，腿2膝关节施加的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain — 成功到达地形末端，属于成功终止。
- failure-like termination: body_fallen_over — 身体摔倒，属于失败终止。
- ambiguous termination: 无。
- truncation: 无显式截断（step 返回 False 作为 truncation 标志）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false — info 字典为空 {}，没有显式 success 标志。
- explicit_failure_flag_available: false — info 字典为空 {}，没有显式 failure 标志。
- allowed_info_fields: 无（info 为空）。
- forbidden_or_uncertain_info_fields: 无（info 为空）。

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，但可通过 horizontal_velocity 积分或 LIDAR 间接推断
- velocity: horizontal_velocity (obs[2])，vertical_velocity (obs[3])
- orientation: hull_angle (obs[0])，hull_angular_velocity (obs[1])
- contact: leg1_contact (obs[8])，leg2_contact (obs[13])
- action/engine: action 本身（4维力矩），可计算动作幅度、变化率
- other: LIDAR 距离 (obs[14..23])，关节角度和速度 (obs[4..7], obs[9..12])

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
| 1 | angle_penalty + angular_vel_penalty + balance_penalty + forward_reward + gate_factor + gated_forward | 166.54 | 166.54 | 0.00 | 820.35 | angle_penalty=-0.034 angular_vel_penalty=-0.000 balance_penalty=-0.034 forward_reward=0.173 gate_factor=0.889 | new_best |
