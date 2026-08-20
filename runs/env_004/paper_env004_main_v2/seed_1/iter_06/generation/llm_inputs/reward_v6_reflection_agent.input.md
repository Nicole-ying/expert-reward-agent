# Search objective
- target_score: 3800.000000
- current_score: 3163.804808
- gap_to_target: 636.195192
- target_achievement_ratio: 83.258%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 3163.804808）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V5 reward: Replace quadratic forward speed bonus with acceleration reward.
    Uses state_to_improvement: reward speed increases (delta-v) rather than
    rewarding the speed state itself, to help the agent escape the moderate-speed
    plateau and discover faster hopping gaits.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress
    w_accel = 1.0        # acceleration bonus for speed increases
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty

    # Main learning signal: forward velocity (clipped to non-negative)
    v_fwd = max(0.0, next_obs[5])
    forward_progress = w_fwd * v_fwd

    # Acceleration bonus: reward increases in forward speed
    # Uses obs[5] (pre-step velocity) and next_obs[5] (post-step velocity)
    delta_v = next_obs[5] - obs[5]
    acceleration_bonus = w_accel * max(0.0, delta_v)

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    total_reward = forward_progress + acceleration_bonus + torso_angle_penalty + torso_angular_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "acceleration_bonus": acceleration_bonus,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty,
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V3 reward: V1 base + quadratic forward speed bonus.
    Restores the successful V1 structure and adds a mild bonus for high forward velocity
    to push the agent beyond a "safe but moderate" speed plateau.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress
    w_fwd2 = 0.1         # quadratic bonus for higher speeds
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty

    # Main learning signal: forward velocity (clipped to non-negative)
    v_fwd = max(0.0, next_obs[5])
    forward_progress = w_fwd * v_fwd

    # Quadratic bonus: disproportionately rewards higher forward speeds
    forward_speed_bonus = w_fwd2 * (v_fwd ** 2)

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    total_reward = forward_progress + forward_speed_bonus + torso_angle_penalty + torso_angular_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "forward_speed_bonus": forward_speed_bonus,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=3163.804808, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[3093.144936, 3216.396881]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 2165.804545 | 92.2% | 92.2% | 99.6% |
| torso_angular_velocity_penalty | -105.646105 | -4.5% | 4.5% | 100.0% |
| torso_angle_penalty | -47.336195 | -2.0% | 2.0% | 100.0% |
| acceleration_bonus | 29.058110 | 1.2% | 1.2% | 46.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个平面单腿铰接体，通过三个关节的连续力矩驱动，使其持续向前跳跃移动。策略必须在保持身体直立、关节力矩高效的前提下，尽可能产生正向的水平位移。跌倒、身体高度异常、躯干倾角过大或状态数值溢出都会导致回合立即结束。奖励设计应鼓励“持续向前推进”，而非仅仅“保持不倒”。

## 3. 观察空间 observation_space
- type: Box
- shape: [11]
- dtype: float32 (假设，通常为float)
- obs[0]: torso_height —— 主体质心垂直高度
- obs[1]: torso_angle —— 躯干相对于垂直方向的倾角
- obs[2]: upper_joint_angle —— 大腿关节角度
- obs[3]: lower_joint_angle —— 小腿关节角度
- obs[4]: foot_joint_angle —— 足部关节角度
- obs[5]: forward_velocity —— 主体水平速度（向前为正）
- obs[6]: vertical_velocity —— 主体垂直速度
- obs[7]: torso_angular_velocity —— 躯干角速度
- obs[8]: upper_joint_speed —— 大腿关节角速度
- obs[9]: lower_joint_speed —— 小腿关节角速度
- obs[10]: foot_joint_speed —— 足部关节角速度

## 4. 动作空间 action_space
- type: Box，连续动作，范围 [-1.0, 1.0]
- action 0: upper_joint_torque —— 大腿关节力矩
- action 1: lower_joint_torque —— 小腿关节力矩
- action 2: foot_joint_torque —— 足部关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止条件；任务目标是持续前进，并不存在“到达终点”的概念，因此没有成功终止。
- failure-like termination: body_height_outside_healthy_range（身体高度超标，可能跌倒或跳太高）、torso_angle_outside_healthy_range（躯干倾角过大，即将倒下）、state_value_outside_finite_healthy_range（观测值出现 NaN 或 inf）均属于失败终止。
- ambiguous termination: 无。
- truncation: time_limit_reached，回合因达到最大步数而截断，既非成功也非失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: **空**（任何 info 字段均不得使用）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_survive, x_position, y_position, z_distance_from_origin 等（所有列出的字段均禁止使用，且 info 实际返回 {}）

## 7. 可用于奖励函数的信号
- position: torso_height（obs[0]）—— 可用于惩罚过高或过低，保持健康高度区间
- velocity: forward_velocity（obs[5]）、vertical_velocity（obs[6]）、torso_angular_velocity（obs[7]）、upper_joint_speed（obs[8]）、lower_joint_speed（obs[9]）、foot_joint_speed（obs[10]）
- orientation: torso_angle（obs[1]）—— 可惩罚偏离直立的角度
- contact: 无直接接触传感器（仅能通过高度和倾角间接推断脚与地面的接触情况）
- action/engine: action[0], action[1], action[2] —— 可惩罚过大的力矩或过快的动作变化

# Compact expert route context
- selected_route_id: locomotion_continuous_control
- recommended_design_roles: forward_progress_reward (前进/速度奖励), terminal_failure_penalty (失败惩罚), energy_penalty (动作能耗惩罚), alive_bonus (存活奖励), action_smoothness_penalty (动作平滑惩罚), stability_penalty (姿态/稳定性惩罚)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_progress + torso_angle_penalty + torso_angular_velocity_penalty | 3168.34 | 3168.34 | 0.00 | 1000.00 | forward_progress=1.712 torso_angle_penalty=-0.040 torso_angular_velocity_penalty=-0.058 | new_best |
| 2 | forward_progress + height_reward + torso_angle_penalty + torso_angular_velocity_penalty | 858.50 | 3168.34 | -2309.85 | 791.50 | forward_progress=0.243 height_reward=0.300 torso_angle_penalty=-0.016 torso_angular_velocity_penalty=-0.022 | no_meaningful_improvement |
| 3 | forward_progress + forward_speed_bonus + torso_angle_penalty + torso_angular_velocity_penalty | 3260.35 | 3260.35 | 0.00 | 1000.00 | forward_progress=1.804 forward_speed_bonus=0.389 torso_angle_penalty=-0.040 torso_angular_velocity_penalty=-0.083 | new_best |
| 4 | forward_progress + forward_speed_bonus + torso_angle_penalty + torso_angular_velocity_penalty | 3038.32 | 3260.35 | -222.03 | 1000.00 | forward_progress=1.829 forward_speed_bonus=0.990 torso_angle_penalty=-0.028 torso_angular_velocity_penalty=-0.054 | unsolved_high_achievement_continue_from_best |
| 5 | acceleration_bonus + forward_progress + torso_angle_penalty + torso_angular_velocity_penalty | 3163.80 | 3260.35 | -96.54 | 1000.00 | acceleration_bonus=0.030 forward_progress=1.872 torso_angle_penalty=-0.040 torso_angular_velocity_penalty=-0.098 | no_meaningful_improvement |
