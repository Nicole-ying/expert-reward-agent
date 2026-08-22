# 1. Search objective
- target_score: 300.000000
- current_score: -53.584149
- gap_to_target: 353.584149
- target_achievement_ratio: -17.861%

# 2. 上一轮奖励函数代码（该轮得分: -53.584149）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract useful signals ----------
    horizontal_speed = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]

    # ---------- health gate: close to 1 when upright, decays when tilting ----------
    denom = 1.0 + 10.0 * hull_angle * hull_angle + 0.1 * hull_angular_velocity * hull_angular_velocity
    health_gate = 1.0 / denom

    # ---------- forward progress ----------
    fwd_speed = max(0.0, horizontal_speed)
    progress_component = 1.0 * fwd_speed * health_gate

    # ---------- action regularisation ----------
    action_sum_sq = action[0]*action[0] + action[1]*action[1] + action[2]*action[2] + action[3]*action[3]
    action_penalty = -0.01 * action_sum_sq

    # ---------- hinge balance penalty: explicit tilt cost beyond safe zone ----------
    tilt_magnitude = abs(hull_angle)
    safe_threshold = 0.4   # ~23 degrees
    excess_tilt = max(0.0, tilt_magnitude - safe_threshold)
    hinge_balance_penalty = -0.5 * excess_tilt

    # ---------- total reward ----------
    total_reward = progress_component + action_penalty + hinge_balance_penalty

    components = {
        "progress": progress_component,
        "action_penalty": action_penalty,
        "hinge_balance_penalty": hinge_balance_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 246.25 | -59.97 | ✅ |
| 2 | 加入双脚离地与垂直速度惩罚后，agent 将学会抑制不安全的腾空行为，保持至少单脚接地，从而减少摔倒、延长存活并最... | 加入双脚离地与垂直速度惩罚后，agent 将学会抑制不安全的腾空行为，保持至少单脚接地，从而减少摔倒、延长存活并最... | 105.95 | -86.30 | ❌ |
| 3 | 骨架变化: air_stability_penalty + balance_penalty + forward_ | — | 74.80 | -95.84 | ❌ |
| 4 | 移除双脚离地惩罚后，agent 可恢复自然摆动相，存活步数回升至 150+，整体 score 接近 -10 ~ 0... | 移除双脚离地惩罚后，agent 可恢复自然摆动相，存活步数回升至 150+，整体 score 接近 -10 ~ 0... | 376.95 | -74.85 | ❓ |
| 5 | 骨架变化: energy_penalty + forward_reward + hinge_penalty | — | 243.15 | -52.46 | ✅ |
| 6 | 让前进奖励的 gate 在身体摇晃时快速衰减， agent 将学会抑制危险振荡，摔倒率下降，有效存活步数和综合得分... | 让前进奖励的 gate 在身体摇晃时快速衰减， agent 将学会抑制危险振荡，摔倒率下降，有效存活步数和综合得分... | 380.95 | -59.50 | ❌ |
| 7 | 加回 hinge_penalty 提供明确的“保持小倾角”梯度，与双因子门控配合，能进一步降低摔倒率，提升平均得分... | 加回 hinge_penalty 提供明确的“保持小倾角”梯度，与双因子门控配合，能进一步降低摔倒率，提升平均得分... | 401.70 | -52.19 | ❓ |
| 8 | 骨架变化: action_penalty + progress | — | 217.05 | -63.34 | ❌ |
| 9 | 显式的倾角惩罚将填补安全梯度缺口，降低摔倒率，恢复存活长度并提升总得分（参考 iter7 有效模式）。 | 显式的倾角惩罚将填补安全梯度缺口，降低摔倒率，恢复存活长度并提升总得分（参考 iter7 有效模式）。 | 216.15 | -53.58 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-53.584149, len=216.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-86.383689, -27.101370]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 70.926116 | 96.4% | 96.4% | 98.4% |
| action_penalty | -2.311209 | -3.1% | 3.1% | 100.0% |
| hinge_balance_penalty | -0.352891 | -0.5% | 0.5% | 1.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
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
| 4 | balance_penalty + forward_reward + terrain_gate + terrain_roughness | -74.85 | -59.97 | -14.88 | 376.95 | balance_penalty=-0.008 forward_reward=0.106 terrain_gate=0.497 terrain_roughness=0.216 | unsolved_stagnation_fresh_restart |
| 5 | energy_penalty + forward_reward + hinge_penalty | -52.46 | -52.46 | 0.00 | 243.15 | energy_penalty=-0.018 forward_reward=0.183 hinge_penalty=-0.001 | new_best |
| 6 | energy_penalty + forward_reward | -59.50 | -52.46 | -7.04 | 380.95 | energy_penalty=-0.018 forward_reward=0.175 | no_meaningful_improvement |
| 7 | energy_penalty + forward_reward + hinge_penalty | -52.19 | -52.19 | 0.00 | 401.70 | energy_penalty=-0.018 forward_reward=0.170 hinge_penalty=-0.004 | unsolved_stagnation_fresh_restart |
| 8 | action_penalty + progress | -63.34 | -52.19 | -11.15 | 217.05 | action_penalty=-0.017 progress=0.187 | no_meaningful_improvement |
| 9 | action_penalty + hinge_balance_penalty + progress | -53.58 | -52.19 | -1.39 | 216.15 | action_penalty=-0.018 hinge_balance_penalty=-0.003 progress=0.182 | no_meaningful_improvement |
