# Search objective
- target_score: 6000.000000
- current_score: -16.602316
- gap_to_target: 6016.602316
- target_achievement_ratio: -0.277%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -16.602316）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：奖励沿前进方向的速度
    w_forward = 1.0
    forward_velocity = next_obs[13]  # 前进速度（主体x方向速度）
    forward_reward = w_forward * forward_velocity

    # 约束1：惩罚横向运动，鼓励直线前进
    w_lateral = 0.5
    lateral_velocity = next_obs[14]  # 横向速度
    lateral_penalty = -w_lateral * (lateral_velocity ** 2)

    # 约束2：惩罚过高的角速度，保持姿态稳定
    w_angular = 0.5
    roll_rate = next_obs[16]
    pitch_rate = next_obs[17]
    yaw_rate = next_obs[18]
    angular_penalty = -w_angular * (roll_rate ** 2 + pitch_rate ** 2 + yaw_rate ** 2)

    # 约束3：惩罚剧烈的垂直运动，防止跳跃/坠落
    w_vertical = 0.1
    vertical_velocity = next_obs[15]  # 垂直速度
    vertical_penalty = -w_vertical * abs(vertical_velocity)

    total_reward = forward_reward + lateral_penalty + angular_penalty + vertical_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "vertical_velocity_penalty": vertical_penalty,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-16.602316, len=11.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-25.541189, -7.891307]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| angular_velocity_penalty | -21.074904 | -71.4% | 71.4% | 100.0% |
| forward_velocity_reward | 0.954357 | 3.2% | 17.1% | 100.0% |
| vertical_velocity_penalty | -2.353788 | -8.0% | 8.0% | 100.0% |
| lateral_velocity_penalty | -1.037845 | -3.5% | 3.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 3D 四足机器人（8 个扭矩控制关节）在平坦地面上稳定地向前行走或奔跑。机器人必须保持身体高度在 0.2~1.0 的范围内；超出该范围（摔倒或蹦飞）或出现数值异常将立即终止回合。核心目标是实现持续、稳定的前进运动，而非仅仅保持站立。

## 3. 观察空间 observation_space
- type: Box
- shape: (105,)
- dtype: float32
- obs[0]: 主体垂直高度 (body_z)
- obs[1]: 主体姿态四元数第 1 分量 (body_orientation)
- obs[2]: 主体姿态四元数第 2 分量
- obs[3]: 主体姿态四元数第 3 分量
- obs[4]: 主体姿态四元数第 4 分量
- obs[5]: 关节 1 角度 (hip_1)
- obs[6]: 关节 2 角度 (ankle_1)
- obs[7]: 关节 3 角度 (hip_2)
- obs[8]: 关节 4 角度 (ankle_2)
- obs[9]: 关节 5 角度 (hip_3)
- obs[10]: 关节 6 角度 (ankle_3)
- obs[11]: 关节 7 角度 (hip_4)
- obs[12]: 关节 8 角度 (ankle_4)
- obs[13]: 主体世界 x 方向速度（前进速度）
- obs[14]: 主体世界 y 方向速度（横向速度）
- obs[15]: 主体垂直速度 (body_z_velocity)
- obs[16]: 主体滚转角速度 (roll)
- obs[17]: 主体俯仰角速度 (pitch)
- obs[18]: 主体偏航角速度 (yaw)
- obs[19]: 关节 1 角速度
- obs[20]: 关节 2 角速度
- obs[21]: 关节 3 角速度
- obs[22]: 关节 4 角速度
- obs[23]: 关节 5 角速度
- obs[24]: 关节 6 角速度
- obs[25]: 关节 7 角速度
- obs[26]: 关节 8 角速度
- obs[27:33]: 躯干接触力/力矩 (6D)
- obs[33:39]: 左前腿上段接触力/力矩
- obs[39:45]: 左前腿下段接触力/力矩
- obs[45:51]: 左前足接触力/力矩（强地面接触信号）
- obs[51:57]: 右前腿上段接触力/力矩
- obs[57:63]: 右前腿下段接触力/力矩
- obs[63:69]: 右前足接触力/力矩（强地面接触信号）
- obs[69:75]: 左后腿上段接触力/力矩
- obs[75:81]: 左后腿下段接触力/力矩
- obs[81:87]: 左后足接触力/力矩（强地面接触信号）
- obs[87:93]: 右后腿上段接触力/力矩
- obs[93:99]: 右后腿下段接触力/力矩
- obs[99:105]: 右后足接触力/力矩（强地面接触信号）

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- dtype: float32
- bounds: [-1.0, 1.0] 每个维度
- action 0: 髋关节 1 力矩 (hip_1_torque)
- action 1: 踝关节 1 力矩 (ankle_1_torque)
- action 2: 髋关节 2 力矩 (hip_2_torque)
- action 3: 踝关节 2 力矩 (ankle_2_torque)
- action 4: 髋关节 3 力矩 (hip_3_torque)
- action 5: 踝关节 3 力矩 (ankle_3_torque)
- action 6: 髋关节 4 力矩 (hip_4_torque)
- action 7: 踝关节 4 力矩 (ankle_4_torque)

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无（本环境没有明确的成功到达目标，只有通过截断结束一个 episode）
- failure-like termination:  
  a) 身体高度超出健康范围 (body_height_outside_healthy_range)，即 body_z ≤ 0.2 或 body_z ≥ 1.0  
  b) 任何状态值出现 NaN 或无穷大 (state_value_outside_finite_range)
- ambiguous termination: 无
- truncation: 到达预设的时间步上限（time_limit_reached），视为正常结束

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典，没有任何允许字段）
- forbidden_or_uncertain_info_fields: 所有与官方奖励相关的字段（如 reward_forward, reward_ctrl, reward_contact, reward_survive）以及 x_position, y_position, distance_from_origin, x_velocity, y_velocity 等均被禁止，不可使用

## 7. 可用于奖励函数的信号
- position: 主体高度 (obs[0])，主体姿态四元数 (obs[1:5])，各关节角度 (obs[5:13])
- velocity: 前进速度 (obs[13])，横向速度 (obs[14])，垂直速度 (obs[15])，滚转/俯仰/偏航角速度 (obs[16:18])，各关节角速度 (obs[19:27])
- orientation: 四元数姿态，可用于惩罚大幅度翻滚或俯仰
- contact: 各足端接触力 (obs[45:51], obs[63:69], obs[81:87], obs[99:105])，可推断足部着地状态与地面反作用力
- action/engine: 关节力矩动作本身 (action[0:8])，可用于惩罚过大的控制量

# Compact expert route context
- selected_route_id: locomotion_continuous_control
- recommended_design_roles: forward_progress_reward (前进/速度奖励), terminal_failure_penalty (失败惩罚), energy_penalty (动作能耗惩罚), alive_bonus (存活奖励), action_smoothness_penalty (动作平滑惩罚), stability_penalty (姿态/稳定性惩罚)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angular_velocity_penalty + forward_velocity_reward + lateral_velocity_penalty + vertical_velocity_penalty | -16.60 | -16.60 | 0.00 | 11.75 | angular_velocity_penalty=-2.262 forward_velocity_reward=0.041 lateral_velocity_penalty=-0.145 vertical_velocity_penalty=-0.076 | new_best |
