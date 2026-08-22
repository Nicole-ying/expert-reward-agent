# Duplicate reward retry
The previous generation duplicated iter 1 (runs/env_005/paper_env005_main_v1/seed_0/iter_01/generation/reward_v1.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from the result state (next_obs)
    body_z = next_obs[0]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]

    # Weights
    w_fwd = 1.0
    w_lat = 0.5
    w_ang = 0.001
    w_h = 10.0

    # 1. Main learning signal: forward velocity reward
    forward_reward = w_fwd * body_x_vel

    # 2. Stability constraint (penalty for dangerous/inefficient deviations)
    lateral_penalty = w_lat * abs(body_y_vel)
    angular_penalty = w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)
    height_penalty = w_h * (
        max(0.0, 0.25 - body_z) + max(0.0, body_z - 0.95)
    )
    stability_penalty = -(lateral_penalty + angular_penalty + height_penalty)

    total_reward = forward_reward + stability_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# Search objective
- target_score: 6000.000000
- current_score: -123.547986
- gap_to_target: 6123.547986
- target_achievement_ratio: -2.059%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -123.547986）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from the result state (next_obs)
    body_z = next_obs[0]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]

    # Weights
    w_fwd = 1.0
    w_lat = 0.5
    w_ang = 0.001
    w_h = 10.0

    # 1. Main learning signal: forward velocity reward
    forward_reward = w_fwd * body_x_vel

    # 2. Stability constraint (penalty for dangerous/inefficient deviations)
    lateral_penalty = w_lat * abs(body_y_vel)
    angular_penalty = w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)
    height_penalty = w_h * (
        max(0.0, 0.25 - body_z) + max(0.0, body_z - 0.95)
    )
    stability_penalty = -(lateral_penalty + angular_penalty + height_penalty)

    total_reward = forward_reward + stability_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-123.547986, len=206.850000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-898.876280, 37.509336]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 81.881057 | 50.7% | 71.9% | 100.0% |
| stability_penalty | -45.380735 | -28.1% | 28.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 3D 四足机器人，通过 8 个扭矩控制关节，实现持续、稳定的前向行走或奔跑，同时保持身体直立。机器人身体高度必须保持在健康范围内（0.2~1.0），过低（倒塌）或过高（弹飞）都会导致回合终止。核心目标是维持稳定的前向运动，而不是仅仅保持站立。

## 3. 观察空间 observation_space
- type: Box
- shape: [105]
- dtype: float
- obs[0]: body_z – 主体垂直位置
- obs[1:5]: body_orientation（四元数，共 4 维）– 主体姿态
- obs[5:13]: joint_angles（共 8 维）– 8 个关节角度
- obs[13]: body_x_velocity – 主体世界 x 轴速度（前向速度）
- obs[14]: body_y_velocity – 主体世界 y 轴速度（横向速度）
- obs[15]: body_z_velocity – 主体垂直速度
- obs[16]: body_roll_velocity – 滚转角速度
- obs[17]: body_pitch_velocity – 俯仰角速度
- obs[18]: body_yaw_velocity – 偏航角速度
- obs[19:27]: joint_velocities（共 8 维）– 8 个关节角速度
- obs[27:33]: torso 接触力/力矩（fx, fy, fz, tx, ty, tz）
- obs[33:39]: front_left upper leg 接触力/力矩
- obs[39:45]: front_left lower leg 接触力/力矩
- obs[45:51]: front_left foot 接触力/力矩
- obs[51:57]: front_right upper leg 接触力/力矩
- obs[57:63]: front_right lower leg 接触力/力矩
- obs[63:69]: front_right foot 接触力/力矩
- obs[69:75]: back_left upper leg 接触力/力矩
- obs[75:81]: back_left lower leg 接触力/力矩
- obs[81:87]: back_left foot 接触力/力矩
- obs[87:93]: back_right upper leg 接触力/力矩
- obs[93:99]: back_right lower leg 接触力/力矩
- obs[99:105]: back_right foot 接触力/力矩

## 4. 动作空间 action_space
- type: Box (continuous)
- shape: [8]
- 范围: [-1.0, 1.0]
- action 0: hip_1_torque – 第一个髋关节扭矩
- action 1: ankle_1_torque – 第一个踝关节扭矩
- action 2: hip_2_torque – 第二个髋关节扭矩
- action 3: ankle_2_torque – 第二个踝关节扭矩
- action 4: hip_3_torque – 第三个髋关节扭矩
- action 5: ankle_3_torque – 第三个踝关节扭矩
- action 6: hip_4_torque – 第四个髋关节扭矩
- action 7: ankle_4_torque – 第四个踝关节扭矩

（注：四足共 8 个驱动关节，对称分布；每腿由髋 + 踝关节组成）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式的成功终止条件；任务要求持续前进，只有时间截断（truncation）可能对应完成整个 episode
- failure-like termination:
  - body_height_outside_healthy_range（主体 z 超出 0.2~1.0）：倒塌或弹飞
  - state_value_outside_finite_range（任何状态量为 NaN 或 inf）
- ambiguous termination: 无
- truncation: time_limit_reached（达到最大步数限制）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []  （环境 step 返回的 info 为空 dict，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin, x_velocity, y_velocity 等均被禁止

## 7. 可用于奖励函数的信号
- position:
  - body_z（obs[0]）– 垂直高度，可用于惩罚离健康范围偏差
- velocity:
  - body_x_velocity（obs[13]）– 前向速度，可直接作为前进奖励
  - body_y_velocity（obs[14]）– 横向速度，可用于抑制侧向漂移
  - body_z_velocity（obs[15]）– 垂直速度，结合高度变化可判断坠落倾向
- orientation:
  - body_orientation 四元数（obs[1:5]）– 可用于保持身体水平，惩罚倾斜
  - roll/pitch/yaw 角速度（obs[16:18]）– 可惩罚快速旋转，维持稳定姿态
- contact:
  - 各腿足部接触力（obs[45:51], [63:69], [81:87], [99:105]）– 判断地面接触，可用于奖励足部触地或惩罚不稳定的接触模式
- action/engine:
  - 关节扭矩 action[0:7] – 可用于惩罚过大扭矩，提高能效
  - 关节速度（obs[19:27]）– 可结合扭矩计算功率

# Compact expert route context
- selected_route_id: locomotion_continuous_control
- recommended_design_roles: forward_progress_reward (前进/速度奖励), terminal_failure_penalty (失败惩罚), energy_penalty (动作能耗惩罚), alive_bonus (存活奖励), action_smoothness_penalty (动作平滑惩罚), stability_penalty (姿态/稳定性惩罚)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_velocity_reward + stability_penalty | -123.55 | -123.55 | 0.00 | 206.85 | forward_velocity_reward=0.234 stability_penalty=-0.240 | new_best |
