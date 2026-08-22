# ⚠️ 上一版代码验证失败
错误信息：Reward v9 failed validation: 出现禁止模式: except  (record: runs/env_005/paper_env005_main_v1/seed_0/iter_09/generation/validations/reward_v9.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Best v6 base with single Level 1 fix: w_height reduced 1.0 -> 0.3.
    Height penalty-to-progress ratio drops from 0.925 to ~0.28, giving forward
    velocity the dominant gradient while retaining a soft height constraint.
    All other components and weights are preserved from v6.
    """

    # ---------- Observation indices (v6, confirmed by best run) ----------
    BODY_X_VEL = 13
    BODY_Y_VEL = 14
    ROLL_VEL   = 16
    PITCH_VEL  = 17

    # Quaternion in (w, x, y, z) order — validated by best run survival
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    body_z     = next_obs[0]
    forward_vel = next_obs[BODY_X_VEL]
    lateral_vel = next_obs[BODY_Y_VEL]
    roll_vel    = next_obs[ROLL_VEL]
    pitch_vel   = next_obs[PITCH_VEL]

    # ---------- Weights (v6 preserved except w_height) ----------
    w_fwd    = 1.0
    w_orient = 0.3
    w_lat    = 0.2
    w_ang    = 0.05
    w_height = 0.3   # only change: 1.0 -> 0.3 (Level 1 scale fix)

    # ---------- Height deviation penalty ----------
    # Healthy range [0.2, 1.0], center 0.6, half-range 0.4
    # Quadratic: gentle near center, steep near boundaries
    height_deviation = (body_z - 0.6) / 0.4
    height_deviation_penalty = -w_height * (height_deviation ** 2)

    # ---------- Forward progress (main learning signal) ----------
    forward_reward = w_fwd * forward_vel

    # ---------- Orientation: keep body upright ----------
    orientation_reward = w_orient * body_up_z

    # ---------- Lateral drift suppression ----------
    lateral_penalty = -w_lat * abs(lateral_vel)

    # ---------- Angular velocity damping ----------
    angular_penalty = -w_ang * (abs(roll_vel) + abs(pitch_vel))

    # ---------- Assemble ----------
    total_reward = (
        forward_reward
        + orientation_reward
        + lateral_penalty
        + angular_penalty
        + height_deviation_penalty
    )

    components = {
        "forward_velocity_reward": forward_reward,
        "orientation_reward": orientation_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "height_deviation_penalty": height_deviation_penalty,
    }

    return float(total_reward), components
```

# Search objective
- target_score: 6000.000000
- current_score: -19.195913
- gap_to_target: 6019.195913
- target_achievement_ratio: -0.320%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -19.195913）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Four-legged locomotion reward v1.
    Core design hypothesis: focus on height management and foot contact rhythm.
    Forward velocity is deliberately absent from v1 to prevent fast-then-fail collapse.
    """

    # ---------- Extract next observation signals ----------
    # Body height
    body_z = next_obs[0]

    # Body orientation (quaternion) — individual indices, no slicing
    quat_w = next_obs[4]  # scalar component
    quat_x = next_obs[1]
    quat_y = next_obs[2]
    quat_z = next_obs[3]

    # Body velocities
    forward_vel = next_obs[13]   # desired forward movement
    lateral_vel = next_obs[14]   # sideways drift
    vertical_vel = next_obs[15]  # bouncing

    # Angular velocities (body rotation)
    roll_rate = next_obs[16]
    pitch_rate = next_obs[17]

    # Foot contact z-force — access with single indices, no slicing allowed
    fl_foot_z = next_obs[47]   # front-left foot fz (contact force z-component)
    fr_foot_z = next_obs[65]   # front-right foot fz
    rl_foot_z = next_obs[83]   # rear-left foot fz
    rr_foot_z = next_obs[101]  # rear-right foot fz

    # ---------- Component 1: Height Management (主学习信号) ----------
    # Role: core driving force — stay standing
    # Mathematical form: symmetric penalty around target height with deadzone
    target_height = 0.6
    height_error = body_z - target_height

    # Symmetric penalty: penalizes both being too low (collapsing) and too high (jumping)
    # Deadzone of +/-0.05 to allow natural body oscillation
    height_deviation = abs(height_error) - 0.05
    if height_deviation < 0.0:
        height_deviation = 0.0

    # Linear penalty beyond deadzone, capped to avoid explosion
    height_penalty = 4.0 * min(height_deviation, 0.4)
    height_management = -height_penalty

    # ---------- Component 2: Foot Contact Regularization (步态引导) ----------
    # Role: gait regularization — encourage rhythmic foot contact patterns
    # Use sigmoid on contact forces to convert to ~[0,1] contact indicator
    # Then reward having at least two feet on ground (diagonal pairs preferred)

    contact_threshold = 50.0  # minimum contact force to consider "grounded"
    sensitivity = 0.05

    # Simple sigmoid activation for each foot
    fl_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (fl_foot_z - contact_threshold))))
    fr_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (fr_foot_z - contact_threshold))))
    rl_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (rl_foot_z - contact_threshold))))
    rr_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (rr_foot_z - contact_threshold))))

    # Total ground contact (0 = no feet, 4 = all feet on ground)
    total_contact = fl_contact + fr_contact + rl_contact + rr_contact

    # Reward: prefer 2-3 feet on ground (walking), penalize 0-1 (falling) or 4 (standing still)
    # Use Gaussian-like reward centered at 2.5 (dynamic gait)
    contact_target = 2.5
    contact_deviation = abs(total_contact - contact_target)
    contact_reward = 1.5 * (2.718281828 ** (-2.0 * (contact_deviation ** 2)))

    # ---------- Component 3: Body Stability (姿态约束) ----------
    # Role: stability constraint — keep body upright and suppress roll/pitch
    # Compute body "up" vector from quaternion (projection of body z-axis onto world z-axis)
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # Reward closeness to 1.0 (perfectly upright)
    tilt_penalty = 1.5 * (1.0 - body_up_z)

    # Also penalize angular velocity (roll + pitch rate)
    angular_rate_penalty = 0.3 * (abs(roll_rate) + abs(pitch_rate))

    stability_cost = tilt_penalty + angular_rate_penalty

    # ---------- Component 4: Forward Motivation (弱前进信号) ----------
    # Role: weak forward bias — gentle encouragement to move forward
    forward_encouragement = 0.3 * forward_vel

    # ---------- Component 5: Lateral Drift Penalty (侧向抑制) ----------
    # Role: suppress side-to-side sway
    lateral_penalty = 0.6 * abs(lateral_vel)

    # ---------- Assemble total reward ----------
    total_reward = (
        height_management
        + contact_reward
        - stability_cost
        + forward_encouragement
        - lateral_penalty
    )

    # ---------- Components dictionary ----------
    components = {
        'height_management': height_management,
        'contact_reward': contact_reward,
        'stability_cost': -stability_cost,
        'forward_encouragement': forward_encouragement,
        'lateral_penalty': -lateral_penalty
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Materially different reward design (v2) — adds height deviation penalty
    to prevent body collapse, which caused 19/20 early terminations in v1.
    All v1 components and weights are preserved unchanged.
    """
    # next_obs indices
    BODY_X_VEL = 13
    BODY_Y_VEL = 14
    ROLL_VEL = 16
    PITCH_VEL = 17

    # Quaternion components: assumed order (w, x, y, z)
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # Body height (vertical position)
    body_z = next_obs[0]

    forward_vel = next_obs[BODY_X_VEL]
    lateral_vel = next_obs[BODY_Y_VEL]
    roll_vel = next_obs[ROLL_VEL]
    pitch_vel = next_obs[PITCH_VEL]

    # Weights — v1 weights preserved
    w_fwd    = 1.0
    w_orient = 0.3
    w_lat    = 0.2
    w_ang    = 0.05

    # New height deviation penalty
    # Healthy range: [0.2, 1.0], center = 0.6, half-range = 0.4
    # At body_z=0.2 or 1.0, penalty = -1.0; at center 0.6, penalty = 0
    w_height = 1.0
    height_deviation = (body_z - 0.6) / 0.4
    height_deviation_penalty = -w_height * (height_deviation ** 2)

    # Components (v1 preserved)
    forward_reward = w_fwd * forward_vel
    orientation_reward = w_orient * body_up_z
    lateral_penalty = -w_lat * abs(lateral_vel)
    angular_penalty = -w_ang * (abs(roll_vel) + abs(pitch_vel))

    total_reward = (
        forward_reward
        + orientation_reward
        + lateral_penalty
        + angular_penalty
        + height_deviation_penalty
    )

    components = {
        "forward_velocity_reward": forward_reward,
        "orientation_reward": orientation_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "height_deviation_penalty": height_deviation_penalty,
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-19.195913, len=10.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-48.286113, 3.325283]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_cost | -35.691970 | -76.1% | 76.1% | 100.0% |
| height_management | -5.710450 | -12.2% | 12.2% | 88.6% |
| lateral_penalty | -3.620356 | -7.7% | 7.7% | 100.0% |
| forward_encouragement | -0.286118 | -0.6% | 4.0% | 100.0% |
| contact_reward | 0.000979 | 0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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