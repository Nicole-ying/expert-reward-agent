# 1. Search objective
- target_score: 300.000000
- current_score: -95.842155
- gap_to_target: 395.842155
- target_achievement_ratio: -31.947%

# 2. 上一轮奖励函数代码（该轮得分: -95.842155）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    hull_angle = obs[0]
    hull_angvel = obs[1]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]
    leg1_contact = obs[12]
    leg2_contact = obs[13]

    # ------------------------------------------------------------
    # 1. Terrain awareness gate (NEW - replaces raw forward_progress)
    #    Use lidar readings (obs[14] to obs[23]) to measure terrain roughness.
    #    Explicit single-index access to avoid any slice syntax.
    # ------------------------------------------------------------
    lidar_0 = obs[14]
    lidar_1 = obs[15]
    lidar_2 = obs[16]
    lidar_3 = obs[17]
    lidar_4 = obs[18]
    lidar_5 = obs[19]
    lidar_6 = obs[20]
    lidar_7 = obs[21]
    lidar_8 = obs[22]
    lidar_9 = obs[23]

    n_lidar = 10.0
    mean_lidar = (
        lidar_0 + lidar_1 + lidar_2 + lidar_3 + lidar_4 +
        lidar_5 + lidar_6 + lidar_7 + lidar_8 + lidar_9
    ) / n_lidar

    variance = (
        (lidar_0 - mean_lidar) ** 2 +
        (lidar_1 - mean_lidar) ** 2 +
        (lidar_2 - mean_lidar) ** 2 +
        (lidar_3 - mean_lidar) ** 2 +
        (lidar_4 - mean_lidar) ** 2 +
        (lidar_5 - mean_lidar) ** 2 +
        (lidar_6 - mean_lidar) ** 2 +
        (lidar_7 - mean_lidar) ** 2 +
        (lidar_8 - mean_lidar) ** 2 +
        (lidar_9 - mean_lidar) ** 2
    ) / n_lidar

    roughness = variance ** 0.5

    # Map roughness to a gate: 1.0 (smooth) -> 0.3 (rough)
    roughness_threshold = 0.3
    roughness_clipped = roughness
    if roughness_clipped > roughness_threshold:
        roughness_clipped = roughness_threshold
    gate = 1.0 - 0.7 * (roughness_clipped / roughness_threshold)  # in [0.3, 1.0]

    forward_reward = horizontal_speed * gate

    # ------------------------------------------------------------
    # 2. Balance penalty (unchanged)
    # ------------------------------------------------------------
    angle_threshold = 0.4
    angvel_threshold = 1.0
    angle_excess = abs(hull_angle) - angle_threshold
    if angle_excess < 0.0:
        angle_excess = 0.0
    angvel_excess = abs(hull_angvel) - angvel_threshold
    if angvel_excess < 0.0:
        angvel_excess = 0.0
    balance_penalty = -3.0 * (angle_excess ** 2) - 0.1 * (angvel_excess ** 2)

    # ------------------------------------------------------------
    # 3. Air-stability penalty (unchanged)
    # ------------------------------------------------------------
    sum_contact = leg1_contact + leg2_contact
    both_feet_off = 1.0 - sum_contact
    if both_feet_off < 0.0:
        both_feet_off = 0.0
    air_penalty = -0.3 * both_feet_off

    neg_vertical_speed = -vertical_speed
    if neg_vertical_speed < 0.0:
        neg_vertical_speed = 0.0
    vertical_fall_penalty = -1.0 * both_feet_off * neg_vertical_speed

    air_stability_penalty = air_penalty + vertical_fall_penalty

    total_reward = forward_reward + balance_penalty + air_stability_penalty

    components = {
        'forward_reward': forward_reward,
        'balance_penalty': balance_penalty,
        'air_stability_penalty': air_stability_penalty,
        'terrain_roughness': roughness,
        'terrain_gate': gate
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 246.25 | -59.97 | ✅ |
| 2 | 加入双脚离地与垂直速度惩罚后，agent 将学会抑制不安全的腾空行为，保持至少单脚接地，从而减少摔倒、延长存活并最... | 加入双脚离地与垂直速度惩罚后，agent 将学会抑制不安全的腾空行为，保持至少单脚接地，从而减少摔倒、延长存活并最... | 105.95 | -86.30 | ❌ |
| 3 | 骨架变化: air_stability_penalty + balance_penalty + forward_ | — | 74.80 | -95.84 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-95.842155, len=74.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-96.019857, -95.640137]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terrain_gate | 37.423793 | 61.9% | 61.9% | 100.0% |
| terrain_roughness | 16.018375 | 26.5% | 26.5% | 100.0% |
| forward_reward | 6.412768 | 10.6% | 10.7% | 100.0% |
| air_stability_penalty | -0.500731 | -0.8% | 0.8% | 13.3% |
| balance_penalty | -0.058223 | -0.1% | 0.1% | 28.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
一个双足机器人在崎岖不平、布满障碍（阶梯、树桩、坑洼）的地形上尽可能远且高效地向前行走。机器人配备前向激光雷达，可感知前方地形高度。核心目标是学会利用激光雷达信息调整步态，在不摔倒的前提下持续前进；附属目标是减少不必要的关节扭矩（能量效率）并争取到达地形终点。不应将“到达终点”误解为唯一成功信号——能够稳定行走不摔倒才是关键，终点到达是终止条件之一但无独立奖励标注。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: 连续浮点 + 部分二值
- 详细字段（按索引）：
  - obs[0]: hull_angle (name: hull_angle) — 身体俯仰/倾斜角，reward_usable: true，用于检测摔倒和姿态稳定
  - obs[1]: hull_angular_velocity — 身体角速度，reward_usable: true，辅助姿态惩罚
  - obs[2]: horizontal_speed — 质心水平速度，reward_usable: true，核心前进信号
  - obs[3]: vertical_speed — 质心垂直速度，reward_usable: true，可能帮助判断弹跳或摔倒
  - obs[4]: hip_1_angle — 第1髋关节角度，reward_usable: true（关节状态跟踪）
  - obs[5]: hip_1_speed — 第1髋关节角速度
  - obs[6]: knee_1_angle — 第1膝关节角度
  - obs[7]: knee_1_speed — 第1膝关节角速度
  - obs[8]: hip_2_angle — 第2髋关节角度
  - obs[9]: hip_2_speed — 第2髋关节角速度
  - obs[10]: knee_2_angle — 第2膝关节角度
  - obs[11]: knee_2_speed — 第2膝关节角速度
  - obs[12]: leg_1_ground_contact — 第1腿接地指示（0/1），reward_usable: true，可作为步态接触约束
  - obs[13]: leg_2_ground_contact — 第2腿接地指示，同上
  - obs[14]~[23]: lidar_1~lidar_10 — 10个激光测距仪读数，表示前方地形高度。reward_usable: 谨慎使用，不可直接作为奖励项，但可间接推导预见性调整；初始训练阶段不建议直接奖励，但可帮助分析失败模式。

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- bounds: [-1.0, 1.0] 连续值
- 动作含义：
  - action[0]: hip_1_torque — 施加到第1髋关节的扭矩
  - action[1]: knee_1_torque — 施加到第1膝关节的扭矩
  - action[2]: hip_2_torque — 施加到第2髋关节的扭矩
  - action[3]: knee_2_torque — 施加到第2膝关节的扭矩

所有动作均为连续扭矩控制，无离散动作。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形尽头），但无显式成功标志。可视为**成功的行走存活**导致的终止。
- failure-like termination: body_fallen_over（身体摔倒），常见于 hull_angle 过大或质心跳跃、触地异常。
- ambiguous termination: 无。
- truncation: 未定义明确截断（step source 中仅 terminated，无 truncated 分支）。因此所有 episode 结束均由终止条件触发。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无 success 标志）
- explicit_failure_flag_available: false（同上）
- allowed_info_fields: []（interface 规定 info_is_empty，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为环境实际不提供任何 info。

尽管如此，可**从观测推导**终止类型：
- 摔倒 (derived_possible): 身体倾斜角 |hull_angle| 超出临界阈值（如 >0.5 rad），或 hull_angular_velocity 突变，同时 leg contact 可能消失。
- 到达终点 (derived_possible): 水平速度仍较高、姿态稳定时 episode 突然终止；也可结合上一步位置推断（但观测无位置），只能依赖速度与姿态平滑终止时的表现进行事后推测。但其可靠性不足以成为奖励条件，可偶尔用于事后分析。

## 7. 可用于奖励函数的信号
- position: 观察中无绝对位置，仅可通过速度累积间接推断位移；无直接可用位置坐标。
- velocity: horizontal_speed（obs[2]）、vertical_speed（obs[3]）、各关节速度（obs[5,7,9,11]）
- orientation: hull_angle（obs[0]）、hull_angular_velocity（obs[1]）
- contact: leg_1_ground_contact（obs[12]）、leg_2_ground_contact（obs[13]）
- action/engine: 动作本身（4维扭矩）
- other:
  - laser scan（obs[14:23]）——可用于推断地形粗糙度，但需谨慎映射为奖励时容易引入噪声；暂时建议不作为常规奖励信号。
  - derived_possible: 通过 hull_angle 阈值或角速度突变推断摔倒；通过 episode 终止时水平速度 & 姿态推断“疑似成功到达”。

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
| 1 | balance_penalty + forward_progress | -59.97 | -59.97 | 0.00 | 246.25 | balance_penalty=-0.011 forward_progress=0.185 | new_best |
| 2 | air_stability_penalty + balance_penalty + forward_progress | -86.30 | -59.97 | -26.33 | 105.95 | air_stability_penalty=-0.142 balance_penalty=-0.011 forward_progress=0.215 | no_meaningful_improvement |
| 3 | air_stability_penalty + balance_penalty + forward_reward + terrain_gate + terrain_roughness | -95.84 | -59.97 | -35.87 | 74.80 | air_stability_penalty=-0.075 balance_penalty=-0.005 forward_reward=0.072 terrain_gate=0.501 terrain_roughness=0.214 | no_meaningful_improvement |
