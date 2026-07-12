# 1. Search objective
- target_score: 300.000000
- current_score: -49.881089
- gap_to_target: 349.881089
- target_achievement_ratio: -16.627%

# 2. 上一轮奖励函数代码（该轮得分: -49.881089）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号（只奖励正向水平速度）
    #    w_up=2.0，凸化平方，拒绝负速度
    # ============================================================
    horizontal_vel = obs[2]
    forward_progress = 2.0 * (max(0.0, horizontal_vel) ** 2)

    # ============================================================
    # 2. gait_quality gate: 步态交替质量作为乘性门控
    #    使用 leg1_contact，leg2_contact 计算交替程度
    #    值域 [0,1]，1 表示两脚接触状态完全相反
    # ============================================================
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    gait_quality = 1.0 - abs(leg1_contact - leg2_contact)

    # 将步态质量作为 0.5~1.0 的乘性因子乘到前进奖励上
    forward_with_gait = forward_progress * (0.5 + 0.5 * gait_quality)

    # ============================================================
    # 3. balance_maintenance: 防摔倒软约束（保持原样）
    #    轻微惩罚倾斜角与角速度
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    balance_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_vel ** 2)

    # ============================================================
    # 4. both_off_ground_penalty: 防止跳跃/双腿腾空（保持原样）
    # ============================================================
    contact_sum = leg1_contact + leg2_contact
    both_off_ground_penalty = -0.2 * (1.0 - contact_sum) if contact_sum < 0.5 else 0.0

    # 组合总奖励（不再包含独立的交替奖励）
    total_reward = forward_with_gait + balance_penalty + both_off_ground_penalty

    components = {
        "forward_with_gait": forward_with_gait,
        "balance_penalty": balance_penalty,
        "both_off_ground_penalty": both_off_ground_penalty,
        "gait_quality": gait_quality,           # 纯诊断，不影响奖励
        "forward_progress_raw": forward_progress # 纯诊断
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 0.00 | -65.96 | ❓ |
| 2 | 将交替信号转为前进的门控乘性因子，消除原地踏步独立收益，迫使机器人通过实际前进获取奖励，同时保持平衡约束。 | 将交替信号转为前进的门控乘性因子，消除原地踏步独立收益，迫使机器人通过实际前进获取奖励，同时保持平衡约束。 | 0.00 | -49.88 | ❓ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=-49.881089, len=194.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-79.360974, -13.406592]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gait_quality | 113.150000 | 35.1% | 35.1% | 58.2% |
| forward_progress_raw | 102.945381 | 32.0% | 32.0% | 96.5% |
| forward_with_gait | 85.639802 | 26.6% | 26.6% | 96.5% |
| both_off_ground_penalty | -15.450000 | -4.8% | 4.8% | 39.7% |
| balance_penalty | -4.946417 | -1.5% | 1.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
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
| 2 | balance_penalty + both_off_ground_penalty + forward_progress_raw + forward_with_gait + gait_quality | -49.88 | -49.88 | 0.00 | 194.45 | balance_penalty=-0.021 both_off_ground_penalty=-0.062 forward_progress_raw=0.234 forward_with_gait=0.191 gait_quality=0.535 | new_best |
