# 1. Search objective
- target_score: 300.000000
- current_score: 297.676687
- gap_to_target: 2.323313
- target_achievement_ratio: 99.226%

# 2. 上一轮奖励函数代码（该轮得分: 297.676687）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    horizontal_vel = obs[2]
    hull_angle = obs[0]

    # ============================================================
    # Component 1: forward_progress (main learning signal)
    # ============================================================
    forward_reward = 1.0 * horizontal_vel

    # ============================================================
    # Component 2: soft_health_gate (safety: attenuates forward reward
    #   when balance degrades; gate = 1.0 for |angle| ≤ 0.25 rad)
    # ============================================================
    gate_lower = 0.25
    gate_upper = 0.5
    gate_raw = max(0.0, 1.0 - (abs(hull_angle) - gate_lower) / (gate_upper - gate_lower))
    gate_factor = gate_raw
    gated_forward = forward_reward * gate_factor

    # ============================================================
    # Component 3: energy_penalty (scaling fix: coefficient -0.02)
    #   Reduction from -0.2 to -0.02 ensures per-step penalty ~0.006
    #   which is only ~3% of main signal, restoring action exploration.
    # ============================================================
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.02 * action_power

    # ============================================================
    # Total reward
    # ============================================================
    total_reward = gated_forward + energy_penalty

    # ============================================================
    # Components dict
    # ============================================================
    components = {
        'forward_reward': forward_reward,
        'gate_factor': gate_factor,
        'gated_forward': gated_forward,
        'energy_penalty': energy_penalty,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 820.35 | 166.54 | ✅ |
| 2 | 将 `angle_penalty` 改为 hinge 并在 `gate_factor` 中引入更宽的免衰减区，可以... | 将 `angle_penalty` 改为 hinge 并在 `gate_factor` 中引入更宽的免衰减区，可以... | 901.50 | 287.89 | ✅ |
| 3 | 加入动作幅度惩罚会迫使 agent 探索更节能的步态，从而拉动外部得分中能量项的提升，逼近 300 分 | 加入动作幅度惩罚会迫使 agent 探索更节能的步态，从而拉动外部得分中能量项的提升，逼近 300 分 | 86.10 | -82.07 | ❌ |
| 4 | 将能量惩罚系数降至 -0.02 可使前进速度信号重新主导梯度，agent 恢复有效步态探索，len 和 score... | 将能量惩罚系数降至 -0.02 可使前进速度信号重新主导梯度，agent 恢复有效步态探索，len 和 score... | 1032.25 | 297.68 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=297.676687, len=1032.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[101.258632, 308.601155]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 1956.536958 | 57.3% | 57.3% | 100.0% |
| gated_forward | 939.183820 | 27.5% | 27.5% | 100.0% |
| forward_reward | 495.405691 | 14.5% | 14.5% | 100.0% |
| energy_penalty | -26.007868 | -0.8% | 0.8% | 100.0% |

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
| 2 | angle_deviation + angle_penalty + angular_vel_penalty + balance_penalty + forward_reward + gate_factor | 287.89 | 287.89 | 0.00 | 901.50 | angle_deviation=0.005 angle_penalty=-0.020 angular_vel_penalty=-0.000 balance_penalty=-0.021 forward_reward=0.236 | new_best |
| 3 | energy_penalty + forward_reward + gate_factor + gated_forward | -82.07 | 287.89 | -369.95 | 86.10 | energy_penalty=-0.178 forward_reward=0.224 gate_factor=1.396 gated_forward=0.352 | no_meaningful_improvement |
| 4 | energy_penalty + forward_reward + gate_factor + gated_forward | 297.68 | 297.68 | 0.00 | 1032.25 | energy_penalty=-0.036 forward_reward=0.304 gate_factor=1.610 gated_forward=0.528 | new_best |
