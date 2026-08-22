# Search objective
- target_score: 3800.000000
- current_score: 1647.491412
- gap_to_target: 2152.508588
- target_achievement_ratio: 43.355%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 1647.491412）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    torso_angle = next_obs[1]
    forward_vel = next_obs[5]
    torso_ang_vel = next_obs[7]

    # 主信号：独立奖励前进速度，不再被姿态因子乘性压制
    forward_reward = forward_vel

    # 姿态约束：仅在倾斜超出阈值后才惩罚，允许跳跃必需的动态倾斜
    tilt_threshold = 0.3
    excess_tilt = max(0.0, abs(torso_angle) - tilt_threshold)
    posture_penalty = 1.0 * (excess_tilt ** 2)

    # 角速度约束：抑制快速翻滚，与历史最佳相同的线性形式
    ang_vel_penalty = 0.05 * abs(torso_ang_vel)

    total_reward = forward_reward - posture_penalty - ang_vel_penalty

    components = {
        'forward_reward': forward_reward,
        'posture_penalty': -posture_penalty,
        'ang_vel_penalty': -ang_vel_penalty
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取关键信号
    torso_angle = next_obs[1]          # 躯干倾角
    forward_vel = next_obs[5]          # 前进速度（水平速度）
    torso_ang_vel = next_obs[7]        # 躯干角速度

    # 姿态因子：躯干越直立（角度接近0），因子越接近1
    k_angle = 10.0
    posture_factor = 2.718281828 ** (-k_angle * (torso_angle ** 2))

    # 主学习信号：前进速度 × 姿态因子
    #   - 前进速度为正则奖励，为负则惩罚，引导 agent 持续向前
    #   - 姿态因子在躯干倾斜时自动衰减，抑制为追求速度而过度倾斜
    speed_posture_reward = forward_vel * posture_factor

    # 轻量稳定约束：惩罚躯干角速度，抑制快速晃动/摔倒
    w_ang_vel = 0.05
    ang_vel_penalty = w_ang_vel * abs(torso_ang_vel)

    total_reward = speed_posture_reward - ang_vel_penalty

    components = {
        'speed_posture_reward': speed_posture_reward,
        'ang_vel_penalty': ang_vel_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=1647.491412, len=656.050000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[295.609801, 2630.356416]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 993.442217 | 98.9% | 98.9% | 100.0% |
| ang_vel_penalty | -11.534522 | -1.1% | 1.1% | 100.0% |
| posture_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个平面内的单腿刚体机器人反复向前跳跃，保持躯干直立，同时合理使用关节力矩。核心目标是实现持续的向前移动（前进速度），而不是仅仅保持平衡。如果身体的任何状态超出健康范围（如躯干高度、倾角或状态值异常），回合将终止。截断条件为达到时间上限。

## 3. 观察空间 observation_space
- type: Box
- shape: (11,)
- dtype: 连续浮点数（由环境决定，通常为 float32 或 float64）
- obs[0]: 躯干高度（torso_height），垂直位置
- obs[1]: 躯干倾角（torso_angle），主身体的朝向角度
- obs[2]: 上腿关节角度（upper_joint_angle）
- obs[3]: 下腿关节角度（lower_joint_angle）
- obs[4]: 脚关节角度（foot_joint_angle）
- obs[5]: 前进速度（forward_velocity），身体的水平速度
- obs[6]: 垂直速度（vertical_velocity），身体的垂直速度
- obs[7]: 躯干角速度（torso_angular_velocity）
- obs[8]: 上腿关节角速度（upper_joint_speed）
- obs[9]: 下腿关节角速度（lower_joint_speed）
- obs[10]: 脚关节角速度（foot_joint_speed）

## 4. 动作空间 action_space
- type: Box (连续)
- 动作维度: 3，每个动作范围 [-1.0, 1.0]
- action 0: 上腿关节扭矩（upper_joint_torque）
- action 1: 下腿关节扭矩（lower_joint_torque）
- action 2: 脚关节扭矩（foot_joint_torque）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确的成功终止信号，该任务一般通过截断表示存活到最大步数。
- failure-like termination: 以下任一条件触发立即终止，且通常视为摔倒或失控：
  - 躯干高度超出健康范围
  - 躯干倾角超出健康范围
  - 状态值超出有限健康范围（如 NaN、无穷大）
- ambiguous termination: 无。终止均源于明确的不健康状态。
- truncation: 达到预设时间步上限，回合被截断。可能表示机器人存活了完整周期，但无法直接用于奖励设计（需要根据观测自行判断性能）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（allowed_info_fields 为空）
- forbidden_or_uncertain_info_fields:
  - reward_forward、reward_ctrl、reward_survive（官方奖励分量）
  - x_position、y_position、z_distance_from_origin（全局位置）
  所有这些字段均禁止在奖励函数中使用。info 实际返回的字典为空（{}）。

## 7. 可用于奖励函数的信号
下列信号均可从 obs、next_obs 或 action 中直接获取，且未被禁止：
- 位置相关：
  - 躯干垂直高度（obs[0]）
  - 躯干倾角（obs[1]）
- 速度相关：
  - 前进速度（obs[5]），水平速度，是核心前进指标
  - 垂直速度（obs[6]）
  - 躯干角速度（obs[7]）
- 关节状态：
  - 关节角度（obs[2], obs[3], obs[4]）
  - 关节角速度（obs[8], obs[9], obs[10]）
- 动作/能耗：
  - 施加的关节扭矩（action[0], action[1], action[2]），可用于惩罚过大动作
- 接触信号：未提供

# Compact expert route context
- selected_route_id: locomotion_continuous_control
- recommended_design_roles: forward_progress_reward (前进/速度奖励), terminal_failure_penalty (失败惩罚), energy_penalty (动作能耗惩罚), alive_bonus (存活奖励), action_smoothness_penalty (动作平滑惩罚), stability_penalty (姿态/稳定性惩罚)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | ang_vel_penalty + speed_posture_reward | 2926.19 | 2926.19 | 0.00 | 1000.00 | ang_vel_penalty=0.026 speed_posture_reward=1.472 | new_best |
| 2 | ang_vel_penalty + speed_posture_reward | 2606.95 | 2926.19 | -319.24 | 977.25 | ang_vel_penalty=0.023 speed_posture_reward=1.568 | no_meaningful_improvement |
| 3 | ang_vel_penalty + speed_posture_reward | 2509.87 | 2926.19 | -416.32 | 828.05 | ang_vel_penalty=0.028 speed_posture_reward=1.508 | no_meaningful_improvement |
| 4 | ang_vel_penalty + energy_penalty + speed_posture_reward | 724.24 | 2926.19 | -2201.95 | 237.15 | ang_vel_penalty=0.024 energy_penalty=0.021 speed_posture_reward=1.427 | unsolved_high_achievement_continue_from_best |
| 5 | ang_vel_penalty + speed_posture_reward + vertical_reward | 2080.00 | 2926.19 | -846.18 | 889.25 | ang_vel_penalty=0.014 speed_posture_reward=1.382 vertical_reward=0.057 | no_meaningful_improvement |
| 6 | acceleration_reward + ang_vel_penalty + posture_penalty + vertical_penalty | 142.50 | 2926.19 | -2783.69 | 71.35 | acceleration_reward=0.080 ang_vel_penalty=-0.004 posture_penalty=-0.001 vertical_penalty=-0.009 | no_meaningful_improvement |
| 7 | ang_vel_penalty + forward_reward + posture_penalty | 1647.49 | 2926.19 | -1278.70 | 656.05 | ang_vel_penalty=-0.026 forward_reward=1.415 posture_penalty=0.000 | unsolved_high_achievement_continue_from_best |
