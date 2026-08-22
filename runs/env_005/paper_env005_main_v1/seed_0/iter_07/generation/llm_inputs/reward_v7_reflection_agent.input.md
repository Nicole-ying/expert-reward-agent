# ⚠️ 上一版代码验证失败
错误信息：Reward v7 failed validation: 出现禁止模式: except  (record: runs/env_005/paper_env005_main_v1/seed_0/iter_07/generation/validations/reward_v7.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
1. `evidence`：iter6 新增 height_deviation_penalty 后 score 微升 (11.05→12.31) 但 episode_length 骤降 (252→161)，全部 20 轮均终止；orientation_reward 每步均值从 0.073 暴跌至 0.031，forward 仍占 magnitude_share 63.7%，表明高度惩罚干扰了姿态学习，机器人采用激进前冲姿态快速倒地。
2. `behavior_diagnosis`：策略采取 ~1 m/s 前向奔跑，但身体姿态不稳（侧向漂移 ~0.73 m/s、角速度 ~2.5 rad/s），约 161 步后翻倒导致 body_z 越界终止，无一存活到截断。
3. `signal_completeness`：职责类型基本完备（前进、姿态、横向、角速度、高度），但 height_deviation_penalty 的二次型以 0.6 为中心惩罚一切偏离，抑制了自然步态的高度振荡，反而削弱了姿态学习，使得稳定信号不可达。
4. `selected_level`：Level 2 — height_deviation_penalty 的二次中心型数学形态直接证据有害（新增后 survival 下降 36%、orientation 暴跌 57%），仅调系数无法解决"惩罚正常步态"的结构缺陷。
5. `selected_intervention`：将 height_deviation_penalty 从 `-((body_z-0.6)/0.4)^2` 二次中心型改为 `-w * [max(0, 0.3-body_z) + max(0, body_z-0.9)]` 边界铰链型，仅在危险区 [0.2~0.3] 和 [0.9~1.0] 产生线性惩罚，安全区 [0.3, 0.9] 内完全为零。
6. `falsifiable_hypothesis`：移除安全区内的虚假高度约束后，策略可将学习容量重新分配给 orientation 和 angular 稳定，episode_length 应回升且 orientation_reward 恢复；若 episode_length 持续下降或 orientation 仍低迷，则说明根因不在高度惩罚形态。
7. `expected_next_round`：episode_length 应从 161 回升（目标 >250），terminated 比例可能仍高但应出现部分存活更久的轨迹，orientation_reward 每步均值应从 0.031 回升，height 组件 active_rate 在安全区内应下降（多数步触发为零）。
8. `main_risk`：边界铰链在安全区内梯度为零，若机器人步态自然高度恰好接近 0.3 或 0.9 则缺乏预防梯度，可能在边界附近振荡；线性惩罚在接近终止边界 (0.2/1.0) 时绝对值仍较小 (~0.2/步)，可能不足以阻止快速翻倒。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: Replace quadratic height deviation penalty (centered at 0.6) with
    boundary-hinge penalty that only activates near dangerous body_z boundaries.
    Safe zone [0.3, 0.9] yields zero height penalty, allowing natural gait oscillations.
    All other components and weights preserved from v2.
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

    # Weights — v2 weights preserved except height penalty restructured
    w_fwd    = 1.0
    w_orient = 0.3
    w_lat    = 0.2
    w_ang    = 0.05

    # Height penalty: boundary-hinge form, zero in safe zone [0.3, 0.9]
    # Only penalizes when body_z approaches termination boundaries [0.2, 1.0]
    w_height = 2.0
    height_too_low  = max(0.0, 0.3 - body_z)
    height_too_high = max(0.0, body_z - 0.9)
    height_boundary_penalty = -w_height * (height_too_low + height_too_high)

    # Components (v2 preserved)
    forward_reward = w_fwd * forward_vel
    orientation_reward = w_orient * body_up_z
    lateral_penalty = -w_lat * abs(lateral_vel)
    angular_penalty = -w_ang * (abs(roll_vel) + abs(pitch_vel))

    total_reward = (
        forward_reward
        + orientation_reward
        + lateral_penalty
        + angular_penalty
        + height_boundary_penalty
    )

    components = {
        "forward_velocity_reward": forward_reward,
        "orientation_reward": orientation_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "height_boundary_penalty": height_boundary_penalty,
    }
    return float(total_reward), components
```

# Search objective
- target_score: 6000.000000
- current_score: 12.313339
- gap_to_target: 5987.686661
- target_achievement_ratio: 0.205%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 12.313339）
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
score=12.313339, len=161.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-181.890465, 175.626337]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 158.515021 | 53.4% | 63.7% | 100.0% |
| orientation_reward | 46.309150 | 15.6% | 15.6% | 100.0% |
| lateral_velocity_penalty | -23.634231 | -8.0% | 8.0% | 100.0% |
| angular_velocity_penalty | -20.519881 | -6.9% | 6.9% | 100.0% |
| height_deviation_penalty | -17.426705 | -5.9% | 5.9% | 100.0% |

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