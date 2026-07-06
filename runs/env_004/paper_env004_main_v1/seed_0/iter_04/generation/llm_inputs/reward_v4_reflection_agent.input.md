# 上一轮奖励函数代码（该轮得分: 1338.039623）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for hopping locomotion: forward progress + vertical activity + stability.
    Components:
        forward_stability_reward – forward velocity weighted by torso uprightness
        vertical_activity         – absolute vertical velocity weighted by uprightness (hopping signal)
        stability_penalty         – penalty for torso tilt and angular velocity
    """
    # --- extract relevant signals from next_obs ---
    torso_angle = next_obs[1]          # rad, 0 = upright
    forward_vel = next_obs[5]          # positive = forward
    vertical_vel = next_obs[6]         # vertical velocity (hopping oscillation)
    torso_ang_vel = next_obs[7]        # rad/s

    # --- shared upright gate ---
    temp = 0.3
    upright_factor = 2.718281828 ** (-abs(torso_angle) / temp)

    # --- forward progress with upright gating ---
    forward_stability_reward = forward_vel * upright_factor

    # --- vertical activity: encourage the up-down dynamics of hopping ---
    w_vert = 0.2
    vertical_activity = w_vert * abs(vertical_vel) * upright_factor

    # --- light stability penalty ---
    w_angle = 0.1
    w_ang_vel = 0.01
    stability_penalty = -w_angle * (torso_angle ** 2) - w_ang_vel * (torso_ang_vel ** 2)

    # --- total reward ---
    total_reward = forward_stability_reward + vertical_activity + stability_penalty

    components = {
        "forward_stability_reward": forward_stability_reward,
        "vertical_activity": vertical_activity,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v1 for hopping locomotion: encouraging forward progress while staying upright.
    Components:
        forward_stability_reward  – forward velocity weighted by torso uprightness
        stability_penalty         – penalty for torso tilt and angular velocity
    No terminal success/failure flags available, so no terminal rewards.
    """
    # --- extract relevant signals from next_obs ---
    torso_angle = next_obs[1]          # rad, 0 = upright
    forward_vel = next_obs[5]          # positive = forward
    torso_ang_vel = next_obs[7]        # rad/s

    # --- forward progress with upright gating ---
    # Exponential decay of the forward reward as the torso tilts.
    # Temperature controls how steeply the reward drops with tilt.
    temp = 0.3
    upright_factor = 2.718281828 ** (-abs(torso_angle) / temp)
    forward_stability_reward = forward_vel * upright_factor

    # --- light stability penalty ---
    # Penalise both the torso angle and its angular velocity to suppress
    # large oscillations while still allowing the hopping motion.
    w_angle = 0.1
    w_ang_vel = 0.01
    stability_penalty = -w_angle * (torso_angle ** 2) - w_ang_vel * (torso_ang_vel ** 2)

    # --- total reward ---
    total_reward = forward_stability_reward + stability_penalty

    components = {
        "forward_stability_reward": forward_stability_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=1338.039623, len=444.400000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[1199.694500, 3060.863104]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_stability_reward | 857.993966 | 90.2% | 90.2% | 100.0% |
| vertical_activity | 92.722003 | 9.7% | 9.7% | 100.0% |
| stability_penalty | -0.548786 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个平面单腿关节体连续向前跳跃，同时保持身体直立、各关节在健康范围内。  
策略必须持续稳定地前进，而不仅仅是维持站立。一旦身体高度、躯干角度或任何状态值超出健康范围，回合即终止（失败）。  
任务期望通过关节扭矩输出实现高效、持续的跳跃前进。

## 3. 观察空间 observation_space
- type: Box
- shape: (11,)
- dtype: float32
- obs[0]: torso_height — 主体垂直高度
- obs[1]: torso_angle — 主体朝向角（弧度）
- obs[2]: upper_joint_angle — 上腿关节角度
- obs[3]: lower_joint_angle — 下腿关节角度
- obs[4]: foot_joint_angle — 脚关节角度
- obs[5]: forward_velocity — 主体水平速度（前向为正）
- obs[6]: vertical_velocity — 主体垂直速度
- obs[7]: torso_angular_velocity — 主体角速度
- obs[8]: upper_joint_speed — 上腿关节角速度
- obs[9]: lower_joint_speed — 下腿关节角速度
- obs[10]: foot_joint_speed — 脚关节角速度

## 4. 动作空间 action_space
- type: Box（连续）
- 形状: (3,)
- 范围: [-1.0, 1.0] 每关节扭矩
- action 0: upper_joint_torque — 上腿铰链扭矩
- action 1: lower_joint_torque — 下腿铰链扭矩
- action 2: foot_joint_torque — 脚铰链扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止，任务目标为持续前进，不设成功停止条件。
- failure-like termination: 
  - `body_height_outside_healthy_range`（躯干高度超出健康范围，如摔倒）
  - `torso_angle_outside_healthy_range`（躯干倾角过大，如倾倒）
  - `state_value_outside_finite_healthy_range`（任意状态值出现NaN或无穷）
- ambiguous termination: 无
- truncation: `time_limit_reached`（达到最大步数时截断，并非失败）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 字典为空）
- forbidden_or_uncertain_info_fields:
  - `reward_forward`
  - `reward_ctrl`
  - `reward_survive`
  - `x_position`
  - `y_position`
  - `z_distance_from_origin`
  （上述字段虽然可能存在于真实环境的 info 中，但已被屏蔽并禁止使用）

## 7. 可用于奖励函数的信号
- position: `torso_height`, `torso_angle`（垂直位置、姿态角；注意不含世界x坐标）
- velocity: `forward_velocity`（前向速度，可直接奖励前进）, `vertical_velocity`, `torso_angular_velocity`, 各关节角速度
- orientation: `torso_angle` （衡量直立）
- contact: 无直接接触信号
- action/engine: `upper_joint_torque`, `lower_joint_torque`, `foot_joint_torque`（用于惩罚能耗或大幅动作）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_stability_reward + stability_penalty | 2663.09 | 2663.09 | 0.00 | 1000.00 | forward_stability_reward=1.176 stability_penalty=-0.007 | new_best |
| 2 | forward_stability_reward + stability_penalty | 2663.09 | 2663.09 | 0.00 | 1000.00 | forward_stability_reward=1.176 stability_penalty=-0.007 | no_meaningful_improvement |
| 3 | forward_stability_reward + stability_penalty + vertical_activity | 1338.04 | 2663.09 | -1325.05 | 444.40 | forward_stability_reward=1.583 stability_penalty=-0.001 vertical_activity=0.170 | no_meaningful_improvement |
