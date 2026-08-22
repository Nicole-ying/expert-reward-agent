# 1. Search objective
- target_score: 2000.000000
- current_score: -12.659497
- gap_to_target: 2012.659497
- target_achievement_ratio: -0.633%

# 2. 上一轮奖励函数代码（该轮得分: -12.659497）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next_obs
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]

    # Body uprightness (1.0 = perfectly upright, 0.0 = tilted)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    # Guard against tiny numerical overshoot
    body_up_z = max(0.0, min(1.0, body_up_z))

    # ---------- Forward progress (bounded, only positive velocity) ----------
    vx = max(0.0, body_x_vel)
    forward_reward = vx / (1.0 + vx)          # bounded in [0, 1), strong gradient at low speed

    # ---------- Height safety: hinge penalty near termination boundaries ----------
    height_low_violation  = max(0.0, 0.3 - body_z)
    height_high_violation = max(0.0, body_z - 0.9)
    height_penalty = height_low_violation**2 + height_high_violation**2

    # ---------- Upright posture penalty ----------
    upright_penalty = (1.0 - body_up_z)**2

    # ---------- Lateral slip penalty ----------
    lateral_penalty = abs(body_y_vel)

    # Weights (balanced so that normal walking yields positive total reward)
    w_forward  = 1.0
    w_height   = 5.0
    w_upright  = 5.0
    w_lateral  = 0.5

    total_reward = (w_forward * forward_reward
                    - w_height   * height_penalty
                    - w_upright  * upright_penalty
                    - w_lateral  * lateral_penalty)

    components = {
        "forward_reward": w_forward * forward_reward,
        "height_penalty": w_height * height_penalty,
        "upright_penalty": w_upright * upright_penalty,
        "lateral_penalty": w_lateral * lateral_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 201.55 | 67.71 | ✅ |
| 2 | 骨架变化: forward_gated + height_reward | — | 463.80 | -271.10 | ❌ |
| 3 | 轻量动作惩罚会引导策略减少不必要的力矩幅度，提升步态能效，从而直接提高外部评价分数（score），同时保持生存长度。 | 轻量动作惩罚会引导策略减少不必要的力矩幅度，提升步态能效，从而直接提高外部评价分数（score），同时保持生存长度。 | 369.00 | -112.41 | ❌ |
| 4 | 高度门控乘入主要奖励会使高度调节成为前进优化的必要条件，降低终止率，从而提升外部生存相关分数。 | 高度门控乘入主要奖励会使高度调节成为前进优化的必要条件，降低终止率，从而提升外部生存相关分数。 | 724.55 | -55.54 | ❓ |
| 5 | 骨架变化: forward_reward + height_penalty + lateral_penalty  | — | 39.05 | -5.09 | ❌ |
| 6 | 门控结构（iter 4 验证可延长生存至 724 步）+ 双界保护（补上 iter 4 缺失的上界）+ 移除高系数... | 门控结构（iter 4 验证可延长生存至 724 步）+ 双界保护（补上 iter 4 缺失的上界）+ 移除高系数... | 1000.00 | -37.13 | ❌ |
| 7 | 骨架变化: gated_forward + height_gate + joint_vel_penalty +  | — | 1000.00 | -353.94 | ❓ |
| 8 | 骨架变化: forward_reward + height_penalty + lateral_penalty  | — | 13.25 | -12.66 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-12.659497, len=13.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-51.653454, -4.515459]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 4.844773 | 54.0% | 54.0% | 77.0% |
| lateral_penalty | 3.787663 | 42.2% | 42.2% | 100.0% |
| upright_penalty | 0.192176 | 2.1% | 2.1% | 100.0% |
| height_penalty | 0.142584 | 1.6% | 1.6% | 15.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 3D 四足机器人连续控制任务。机器人拥有四条腿、八个力矩控制关节，需要在保持身体直立且高度处于健康范围的前提下，尽可能稳定地向前行走或奔跑。主要目标是持续、快速的**前进运动**，而非仅仅保持平衡或存活。次要目标可包括维持身体姿态稳定、动作平滑、能量高效，但这些都服务于前进这一核心目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32（推断）
- 维度含义表（索引 0~26）

| obs index | name | meaning | reward_usable |
|---|---|---|---|
| 0 | body_z | 机器人主体的垂直高度 | true |
| 1 | quat_w | 身体方向四元数实部 | true |
| 2 | quat_x | 身体方向四元数虚部 x | true |
| 3 | quat_y | 身体方向四元数虚部 y | true |
| 4 | quat_z | 身体方向四元数虚部 z | true |
| 5 | joint_1_angle | 第 1 髋关节角度 | true |
| 6 | joint_2_angle | 第 1 踝关节角度 | true |
| 7 | joint_3_angle | 第 2 髋关节角度 | true |
| 8 | joint_4_angle | 第 2 踝关节角度 | true |
| 9 | joint_5_angle | 第 3 髋关节角度 | true |
| 10 | joint_6_angle | 第 3 踝关节角度 | true |
| 11 | joint_7_angle | 第 4 髋关节角度 | true |
| 12 | joint_8_angle | 第 4 踝关节角度 | true |
| 13 | body_x_velocity | 世界 x 方向前进速度 | true |
| 14 | body_y_velocity | 世界 y 方向横向速度 | true |
| 15 | body_z_velocity | 垂直速度 | true |
| 16 | body_roll_velocity | 滚转角速度 | true |
| 17 | body_pitch_velocity | 俯仰角速度 | true |
| 18 | body_yaw_velocity | 偏航角速度 | true |
| 19 | joint_1_velocity | 第 1 髋关节角速度 | true |
| 20 | joint_2_velocity | 第 1 踝关节角速度 | true |
| 21 | joint_3_velocity | 第 2 髋关节角速度 | true |
| 22 | joint_4_velocity | 第 2 踝关节角速度 | true |
| 23 | joint_5_velocity | 第 3 髋关节角速度 | true |
| 24 | joint_6_velocity | 第 3 踝关节角速度 | true |
| 25 | joint_7_velocity | 第 4 髋关节角速度 | true |
| 26 | joint_8_velocity | 第 4 踝关节角速度 | true |

额外可用派生：  
- body_up_z = 1 - 2*(quat_x² + quat_y²)，范围 [-1,1]，1 表示完全直立，可用于姿态奖励。
- 所有关节角度、速度可用于动作平滑或关节姿态惩罚。

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- continuous: true
- bounds: [-1.0, 1.0] per joint（对应标准化力矩）

| action dim | name | meaning |
|---|---|---|
| 0 | hip_1_torque | 第 1 髋关节力矩 |
| 1 | ankle_1_torque | 第 1 踝关节力矩 |
| 2 | hip_2_torque | 第 2 髋关节力矩 |
| 3 | ankle_2_torque | 第 2 踝关节力矩 |
| 4 | hip_3_torque | 第 3 髋关节力矩 |
| 5 | ankle_3_torque | 第 3 踝关节力矩 |
| 6 | hip_4_torque | 第 4 髋关节力矩 |
| 7 | ankle_4_torque | 第 4 踝关节力矩 |

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: 无显式成功终止标志。
- **failure-like termination**:  
  - body_height_outside_healthy_range: 主体垂直高度 ≤ 0.2 或 ≥ 1.0 时立即终止（可分别视为摔倒或腾空失控）。  
  - state_value_outside_finite_range: 任意状态值变为 NaN 或无穷大时终止（数值崩溃）。
- **ambiguous termination**:  
  - truncated = time_limit_reached，仅代表时间耗尽，不能直接诠释为成功或失败。
- **truncation**: 由时间限制触发。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false  
  （虽然终止原因可推断为高度越界或数值异常，但在 compute_reward 接口中无法获取 terminated 标志，只能通过 next_obs 的有限信息间接判断。）
- allowed_info_fields: []（本環境不允许在 reward 中使用任何 info 字段）
- forbidden_or_uncertain_info_fields:  
  reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等均不可用。

## 7. 可用于奖励函数的信号
- **position**:  
  - body_z (高度，可做高度保持奖励)  
  - 四元数 → 直立度 body_up_z  
  - 关节角度（可用于姿态正则化）
- **velocity**:  
  - body_x_velocity (世界 x 方向前进速度，核心前进信号)  
  - body_y_velocity, body_z_velocity（横向、垂直速度，可用于惩罚非前进方向运动）
  - 身体角速度 (roll, pitch, yaw) 及关节角速度（可用于平稳性惩罚）
- **orientation**:  
  - 通过四元数计算直立即时状态
- **contact**: 无（此环境无接触力信息）
- **action/engine**:  
  - 8 个关节力矩（可用于动作幅度惩罚、平滑性惩罚）
- **other**:  
  - next_obs 与 obs 的差分可用于瞬时变化量。

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
| 1 | forward_reward + height_reward + upright_reward | 67.71 | 67.71 | 0.00 | 201.55 | forward_reward=0.668 height_reward=-0.034 upright_reward=-0.215 | new_best |
| 2 | forward_gated + height_reward | -271.10 | 67.71 | -338.81 | 463.80 | forward_gated=0.343 height_reward=-0.044 | no_meaningful_improvement |
| 3 | action_penalty + forward_gated + height_reward | -112.41 | 67.71 | -180.12 | 369.00 | action_penalty=-0.044 forward_gated=0.505 height_reward=-0.037 | no_meaningful_improvement |
| 4 | action_penalty + forward_gated_height | -55.54 | 67.71 | -123.25 | 724.55 | action_penalty=-0.043 forward_gated_height=0.462 | unsolved_stagnation_fresh_restart |
| 5 | forward_reward + height_penalty + lateral_penalty + upright_penalty | -5.09 | 67.71 | -72.80 | 39.05 | forward_reward=0.491 height_penalty=-0.022 lateral_penalty=-0.359 upright_penalty=-3.674 | no_meaningful_improvement |
| 6 | gated_forward + height_gate + lateral_penalty + upright_reward | -37.13 | 67.71 | -104.84 | 1000.00 | gated_forward=1.150 height_gate=0.817 lateral_penalty=-0.098 upright_reward=0.029 | no_meaningful_improvement |
| 7 | gated_forward + height_gate + joint_vel_penalty + lateral_penalty + upright_reward | -353.94 | 67.71 | -421.65 | 1000.00 | gated_forward=0.959 height_gate=0.840 joint_vel_penalty=-0.372 lateral_penalty=-0.051 upright_reward=0.039 | unsolved_stagnation_fresh_restart |
| 8 | forward_reward + height_penalty + lateral_penalty + upright_penalty | -12.66 | 67.71 | -80.37 | 13.25 | forward_reward=0.227 height_penalty=0.009 lateral_penalty=0.288 upright_penalty=1.278 | no_meaningful_improvement |
