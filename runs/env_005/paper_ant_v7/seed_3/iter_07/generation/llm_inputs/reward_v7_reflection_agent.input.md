# ⚠️ 上一版代码验证失败
错误信息：Reward v7 failed validation: 出现未允许的 obs/next_obs 切片 (record: runs\env_005\paper_ant_v7\seed_3\iter_07\generation\validations\reward_v7.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由
上一轮骨架基于**高度门控前进速度**，成功将生存长度恢复至满1000步，但外部评分依然为负（-37.13）。观察组件表：主信号 `gated_forward` 每步均值约2.58，但该策略几乎必定伴随高能耗/高力矩的动作模式（八关节高频大力矩），导致环境外部评价中的能耗惩罚扣分远大于前进得分，因此出现“训练奖励很高、外部分数很低”的严重不对齐。  
累积记录中 iter 3 曾直接加入全局动作幅度惩罚，但导致生存长度从724暴跌至369。本轮不重复该错误，改用**轻量关节角速度平方惩罚**作为效率信号：角速度越大表示动作越剧烈、能耗越高，且该惩罚连续、无界但系数极小，不会在安全区域造成灾难性 gate 塌缩。  
此改动为 Level 2 结构变换（新增能耗组件），在不改变主信号骨架的前提下向策略注入“节能”梯度，预期可降低力矩幅度从而提高外部评分，同时维持生存长度。

# 代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract observation indices ----------
    body_z_next = next_obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_vel = obs[13]
    body_y_vel = obs[14]

    # ---------- forward velocity reward (primary) ----------
    w_fwd = 1.5
    forward_reward = w_fwd * body_x_vel

    # ---------- height safety gate (dual-bound) ----------
    # termination: body_z <= 0.2 or body_z >= 1.0
    # safe zone: [0.35, 0.85] where gate = 1.0
    # gate decays linearly to 0 at boundaries [0.2, 1.0]
    z_low_safe = 0.35
    z_low_dead = 0.2
    z_high_safe = 0.85
    z_high_dead = 1.0

    gate_low = min(1.0, max(0.0, (body_z_next - z_low_dead) / (z_low_safe - z_low_dead)))
    gate_high = min(1.0, max(0.0, (z_high_dead - body_z_next) / (z_high_dead - z_high_safe)))
    height_gate = gate_low * gate_high

    # ---------- gated forward ----------
    gated_forward = forward_reward * height_gate

    # ---------- upright posture reward (light) ----------
    # body_up = 1 - 2*(qx^2 + qy^2), ranges [-1, 1], 1 = perfectly upright
    body_up = 1.0 - 2.0 * (qx * qx + qy * qy)
    w_up = 0.1
    upright_reward = w_up * body_up

    # ---------- lateral drift penalty (light) ----------
    w_lat = 0.3
    lateral_penalty = -w_lat * (body_y_vel ** 2)

    # ---------- joint velocity smoothness penalty (new, light) ----------
    # joint velocities: obs[19..26], punishment for high-speed joints promotes energy efficiency
    joint_vels = obs[19:27]
    # sum of squares of joint velocities
    joint_vel_sq_sum = sum(v * v for v in joint_vels)
    w_joint_vel = 0.01   # very light, per-step penalty << 0.77
    joint_vel_penalty = -w_joint_vel * joint_vel_sq_sum

    # ---------- total reward ----------
    total_reward = gated_forward + upright_reward + lateral_penalty + joint_vel_penalty

    components = {
        "gated_forward": gated_forward,
        "height_gate": height_gate,
        "upright_reward": upright_reward,
        "lateral_penalty": lateral_penalty,
        "joint_vel_penalty": joint_vel_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少能量/动作平滑性信号，agent 以高能耗模式前进导致外部评分低
- **behavior**: agent 成功存活满 1000 步并高速前进，但外部评价仍为负（高能耗扣分抵消前进得分）
- **signal**: 缺少对关节剧烈运动的惩罚（效率信号）
- **level**: Level 2
- **hypothesis**: 加入极轻的关节角速度惩罚可在不危及生存的前提下使策略偏好更节能的步态，从而提升外部分数
- **risk**: 若系数过大可能缩短生存长度；当前系数极小（0.01）副作用概率低，但可能不足以显著改善评分，后续可能需微调

# 1. Search objective
- target_score: 2000.000000
- current_score: -37.126816
- gap_to_target: 2037.126816
- target_achievement_ratio: -1.856%

# 2. 上一轮奖励函数代码（该轮得分: -37.126816）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract observation indices ----------
    body_z_next = next_obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_vel = obs[13]
    body_y_vel = obs[14]

    # ---------- forward velocity reward (primary) ----------
    w_fwd = 1.5
    forward_reward = w_fwd * body_x_vel

    # ---------- height safety gate (dual-bound) ----------
    # termination: body_z <= 0.2 or body_z >= 1.0
    # safe zone: [0.35, 0.85] where gate = 1.0
    # gate decays linearly to 0 at boundaries [0.2, 1.0]
    z_low_safe = 0.35
    z_low_dead = 0.2
    z_high_safe = 0.85
    z_high_dead = 1.0

    gate_low = min(1.0, max(0.0, (body_z_next - z_low_dead) / (z_low_safe - z_low_dead)))
    gate_high = min(1.0, max(0.0, (z_high_dead - body_z_next) / (z_high_dead - z_high_safe)))
    height_gate = gate_low * gate_high

    # ---------- gated forward ----------
    gated_forward = forward_reward * height_gate

    # ---------- upright posture reward (light) ----------
    # body_up = 1 - 2*(qx^2 + qy^2), ranges [-1, 1], 1 = perfectly upright
    body_up = 1.0 - 2.0 * (qx * qx + qy * qy)
    w_up = 0.1
    upright_reward = w_up * body_up

    # ---------- lateral drift penalty (light) ----------
    w_lat = 0.3
    lateral_penalty = -w_lat * (body_y_vel ** 2)

    # ---------- total reward ----------
    total_reward = gated_forward + upright_reward + lateral_penalty

    components = {
        "gated_forward": gated_forward,
        "height_gate": height_gate,
        "upright_reward": upright_reward,
        "lateral_penalty": lateral_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-37.126816, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-902.563877, 150.464044]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 2577.877857 | 68.8% | 69.3% | 93.2% |
| height_gate | 944.245138 | 25.2% | 25.2% | 100.0% |
| lateral_penalty | -109.586332 | -2.9% | 2.9% | 93.1% |
| upright_reward | 78.341314 | 2.1% | 2.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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