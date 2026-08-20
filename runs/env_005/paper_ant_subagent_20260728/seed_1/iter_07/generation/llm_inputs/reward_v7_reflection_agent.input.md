# 1. Search objective
- target_score: 2000.000000
- current_score: -231.683034
- gap_to_target: 2231.683034
- target_achievement_ratio: -11.584%

# 2. 上一轮奖励函数代码（该轮得分: -231.683034）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    body_z_velocity = obs[15]

    # ---------- Derived signals ----------
    # Uprightness: 1.0 = perfectly upright, -1.0 = upside-down
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Height gate (identical to iter4, proven effective) ----------
    height_low_danger = max(0.0, (0.30 - body_z) / 0.10)   # 0.30→0.20
    height_low_factor = max(0.0, 1.0 - height_low_danger)
    height_high_danger = max(0.0, (body_z - 0.90) / 0.10)  # 0.90→1.00
    height_high_factor = max(0.0, 1.0 - height_high_danger)
    height_factor = min(height_low_factor, height_high_factor)

    # ---------- Reward components ----------
    # Strong forward propulsion (inherited from iter5)
    forward_reward = 2.0 * body_x_velocity * height_factor

    # Hinge upright penalty — activates only when tilted (body_up_z < 0)
    # Replaces weak upright_bonus with a corrective gradient
    upright_penalty = -0.5 * max(0.0, 0.0 - body_up_z)

    # Light hinge penalties (unchanged from iter5)
    lateral_deviation = abs(body_y_velocity)
    lateral_penalty = -0.1 * max(0.0, lateral_deviation - 0.3)

    vertical_deviation = abs(body_z_velocity)
    vertical_penalty = -0.1 * max(0.0, vertical_deviation - 0.5)

    # ---------- Total reward ----------
    total_reward = forward_reward + upright_penalty + lateral_penalty + vertical_penalty

    components = {
        "forward": forward_reward,
        "upright_penalty": upright_penalty,
        "lateral_penalty": lateral_penalty,
        "vertical_penalty": vertical_penalty,
        "height_factor": height_factor,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 59.15 | -11.32 | ✅ |
| 2 | health_gate 将"保持安全高度和基本直立"从被动挨罚变成获得 forward 奖励的前提，agent 每... | health_gate 将"保持安全高度和基本直立"从被动挨罚变成获得 forward 奖励的前提，agent 每... | 244.40 | -240.56 | ❌ |
| 3 | 骨架变化: forward_gated + health_gate + upright_penalty + z_ | — | 319.60 | -101.97 | ❌ |
| 4 | 给前进始终保留梯度（仅受高度门控），直立用温和奖励引导而非惩罚，消除死亡螺旋，恢复iter 1的前进势头同时延长存活。 | 给前进始终保留梯度（仅受高度门控），直立用温和奖励引导而非惩罚，消除死亡螺旋，恢复iter 1的前进势头同时延长存活。 | 989.80 | 69.24 | ➖ |
| 5 | 加倍前向系数、轻量横向/垂直 hinge 惩罚将打破净零平衡，驱动更快的持续前进，同时保持高度安全，最终使 sco... | 加倍前向系数、轻量横向/垂直 hinge 惩罚将打破净零平衡，驱动更快的持续前进，同时保持高度安全，最终使 sco... | 713.20 | -664.53 | ❌ |
| 6 | 给倾斜施加强梯度会迫使 agent 快速恢复直立，从而使 forward 奖励恢复 iter 4 水平，分数回正。 | 给倾斜施加强梯度会迫使 agent 快速恢复直立，从而使 forward 奖励恢复 iter 4 水平，分数回正。 | 320.35 | -231.68 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-231.683034, len=320.350000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-1129.435872, 31.720068]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward | 380.264009 | 50.1% | 56.3% | 100.0% |
| height_factor | 319.258559 | 42.0% | 42.0% | 100.0% |
| lateral_penalty | -7.940552 | -1.0% | 1.0% | 41.0% |
| vertical_penalty | -4.514129 | -0.6% | 0.6% | 27.4% |
| upright_penalty | -0.011902 | -0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Score=-231.68, 80% termination (16/20), ep_len=320. Generated reward per-step is positive (+0.35) but original_env_reward=-0.91/step. The agent earns ~687 generated reward per episode while the environment kills it. Score range [-1129, +32] — one seed hit positive, most crash hard.

**Component Anomalies**: upright_penalty is dead in final policy (0.0% active, ep_sum=-0.01) despite 53% nonzero during training — the hinge at body_up_z<0 is never crossed. forward dominates at 56% magnitude share. lateral_penalty (41% active) and vertical_penalty (27% active) contribute negligible penalty.

**Training Dynamics**: No temporal snapshots available. Cross-sectional evidence: forward grew from per-step training mean 0.65 to final-policy ep_sum 380 (≈1.19/step). height_factor stable at ~1.0. upright penalty vanished from training (53%→0%), suggesting policy learned to avoid tilt entirely.

**Signal Quality**: Dead gate: upright_penalty hinge at body_up_z=0 is never crossed — no corrective gradient when agent is nearly upright but unstable. Generated reward is decoupled from survival: +0.35/step gen vs -0.91 env. The agent optimizes forward velocity successfully but ignores the termination signal that kills it.

**Evidence Confidence**: `high`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个连续控制四足机器人运动的环境。主要目标是使四足机器人持续、稳定地向前行走或奔跑，同时保持身体在健康高度范围内（0.2 ~ 1.0），并维持身体基本直立。机器人不应仅仅满足于不倒，而是需要产生有效的向前位移。次要目标包括抑制横向滑移、垂直弹跳、过大的关节力矩或速度，以提升运动的平稳性和能效。没有预设的目标位置，成功由存活至时间截断并持续向前移动来体现，不存在显式的“到达终点”成功标志。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32（推断）
- each index 的含义与 reward 可用性：

| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | body_z | 身体质心垂直位置（高度） | true |
| 1 | quat_w | 身体姿态四元数实部 | true（需组合） |
| 2 | quat_x | 身体姿态四元数 x | true |
| 3 | quat_y | 身体姿态四元数 y | true |
| 4 | quat_z | 身体姿态四元数 z | true |
| 5 | joint_1_angle | 第1髋关节角度 | true |
| 6 | joint_2_angle | 第1踝关节角度 | true |
| 7 | joint_3_angle | 第2髋关节角度 | true |
| 8 | joint_4_angle | 第2踝关节角度 | true |
| 9 | joint_5_angle | 第3髋关节角度 | true |
| 10 | joint_6_angle | 第3踝关节角度 | true |
| 11 | joint_7_angle | 第4髋关节角度 | true |
| 12 | joint_8_angle | 第4踝关节角度 | true |
| 13 | body_x_velocity | 身体在世界系 x 轴的线速度（前进速度） | true |
| 14 | body_y_velocity | 身体在世界系 y 轴的线速度（横向速度） | true |
| 15 | body_z_velocity | 身体在 z 轴的速度（垂直速度） | true |
| 16 | body_roll_velocity | 身体滚转角速度 | true |
| 17 | body_pitch_velocity | 身体俯仰角速度 | true |
| 18 | body_yaw_velocity | 身体偏航角速度 | true |
| 19 | joint_1_velocity | 第1髋关节角速度 | true |
| 20 | joint_2_velocity | 第1踝关节角速度 | true |
| 21 | joint_3_velocity | 第2髋关节角速度 | true |
| 22 | joint_4_velocity | 第2踝关节角速度 | true |
| 23 | joint_5_velocity | 第3髋关节角速度 | true |
| 24 | joint_6_velocity | 第3踝关节角速度 | true |
| 25 | joint_7_velocity | 第4髋关节角速度 | true |
| 26 | joint_8_velocity | 第4踝关节角速度 | true |

注：所有观测均可用于奖励，没有禁止字段。四元数可转换为直立度量：`body_up_z = 1 - 2*(quat_x² + quat_y²)`，取值范围大约 `[-1,1]`，`1.0` 表示完美直立。

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- continuous: true
- bounds: [-1.0, 1.0] 每维
- 动作维度含义：

| dim | name | meaning |
|-----|------|---------|
| 0 | hip_1_torque | 第1髋关节力矩 |
| 1 | ankle_1_torque | 第1踝关节力矩 |
| 2 | hip_2_torque | 第2髋关节力矩 |
| 3 | ankle_2_torque | 第2踝关节力矩 |
| 4 | hip_3_torque | 第3髋关节力矩 |
| 5 | ankle_3_torque | 第3踝关节力矩 |
| 6 | hip_4_torque | 第4髋关节力矩 |
| 7 | ankle_4_torque | 第4踝关节力矩 |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无显式成功条件；时间截断（`truncated = time_limit_reached`）且未提前失败时可视为一种存活成功，但从终止信号角度不具有明确成功标志。
- **failure-like termination**: 身体高度超出 (0.2, 1.0) 范围（即跌落或弹起过高）；任意状态值变为 NaN 或 infinity。
- **ambiguous termination**: 无。
- **truncation**: 达到最大时间步，truncated 为 True，此时 episode 正常结束，不应视为失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields:  
  `reward_forward`, `reward_ctrl`, `reward_contact`, `reward_survive`, `x_position`, `y_position`, `distance_from_origin` 等均被明确禁止。  
  说明：虽然原始环境可能提供这些字段，但当前实验环境下已屏蔽，奖励函数中不得使用。

## 7. 可用于奖励函数的信号
- position: 
  - 身体高度 `body_z` (obs[0])
  - 关节角度 (obs[5:13])
- velocity:
  - 前进速度 `body_x_velocity` (obs[13])
  - 横向速度 `body_y_velocity` (obs[14])
  - 垂直速度 `body_z_velocity` (obs[15])
  - 关节速度 (obs[19:27])
- orientation:
  - 四元数 (obs[1:5])，可计算出直立程度 `body_up_z`
  - 角速度 roll/pitch/yaw (obs[16:18])，可额外作用于稳定性
- action/engine:
  - 动作向量 `action` (8,)，可用于惩罚大扭矩、动作变化等
- other:
  - 无。

# 7. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward + height_penalty + upright_penalty | -11.32 | -11.32 | 0.00 | 59.15 | forward=0.617 height_penalty=-0.003 upright_penalty=-1.383 | new_best |
| 2 | forward_gated + health_gate + upright_penalty + z_velocity_penalty | -240.56 | -11.32 | -229.24 | 244.40 | forward_gated=0.259 health_gate=0.515 upright_penalty=-0.367 z_velocity_penalty=-0.004 | no_meaningful_improvement |
| 3 | forward_gated + health_gate + upright_penalty + z_velocity_penalty | -101.97 | -11.32 | -90.65 | 319.60 | forward_gated=0.374 health_gate=0.574 upright_penalty=-0.323 z_velocity_penalty=-0.008 | no_meaningful_improvement |
| 4 | forward + height_factor + upright_bonus | 69.24 | 69.24 | 0.00 | 989.80 | forward=0.641 height_factor=0.874 upright_bonus=0.028 | unsolved_improving_continue_from_best |
| 5 | forward + height_factor + lateral_penalty + upright_bonus + vertical_penalty | -664.53 | 69.24 | -733.77 | 713.20 | forward=0.182 height_factor=0.771 lateral_penalty=-0.017 upright_bonus=-0.073 vertical_penalty=-0.011 | no_meaningful_improvement |
| 6 | forward + height_factor + lateral_penalty + upright_penalty + vertical_penalty | -231.68 | 69.24 | -300.93 | 320.35 | forward=0.650 height_factor=0.849 lateral_penalty=-0.033 upright_penalty=-0.245 vertical_penalty=-0.020 | no_meaningful_improvement |
