# Duplicate reward retry
The previous generation duplicated iter 6 (runs/env_001/paper_lander_default_ppo_v1/seed_1/iter_06/generation/reward_v6.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    观测:
    obs[0]: x_position_relative_to_target
    obs[1]: y_position_relative_to_pad_height
    obs[2]: x_velocity
    obs[3]: y_velocity
    obs[4]: body_angle
    obs[5]: angular_velocity
    obs[6]: left_contact_flag (0.0/1.0)
    obs[7]: right_contact_flag (0.0/1.0)
    """

    def distance(obs_arr):
        return (obs_arr[0] ** 2 + obs_arr[1] ** 2 + 1e-8) ** 0.5

    # ---- 1. Progress reward: 距离减少量 ----
    dist_old = distance(obs)
    dist_new = distance(next_obs)
    progress_reward = dist_old - dist_new

    # ---- 2. Speed tracking reward ----
    max_speed = 5.0
    d_ref = 1.0
    desired_speed = max_speed * min(dist_new / d_ref, 1.0)

    cur_speed = (next_obs[2] ** 2 + next_obs[3] ** 2 + 1e-8) ** 0.5
    speed_error = abs(cur_speed - desired_speed)
    lambda_speed = 0.2
    speed_tracking_reward = -lambda_speed * speed_error

    # ---- 3. Landing improvement: 着陆质量改善量 (state_to_improvement) ----
    proximity_threshold = 0.5
    speed_threshold = 0.25
    angle_threshold = 0.2

    def landing_quality(o):
        d = (o[0] ** 2 + o[1] ** 2 + 1e-8) ** 0.5
        s = (o[2] ** 2 + o[3] ** 2 + 1e-8) ** 0.5
        prox = max(0.0, 1.0 - d / proximity_threshold)
        slow = max(0.0, 1.0 - s / speed_threshold)
        ang = max(0.0, 1.0 - abs(o[4]) / angle_threshold)
        contact = (o[6] + o[7]) * 0.5
        return prox + slow + ang + 2.0 * contact

    Q_old = landing_quality(obs)
    Q_new = landing_quality(next_obs)

    landing_improvement_coeff = 5.0
    landing_improvement = landing_improvement_coeff * max(0.0, Q_new - Q_old)

    # ---- 总奖励 ----
    total_reward = progress_reward + speed_tracking_reward + landing_improvement

    components = {
        'progress_reward': progress_reward,
        'speed_tracking_reward': speed_tracking_reward,
        'landing_improvement': landing_improvement
    }
    return float(total_reward), components
```

# Search objective
- target_score: 200.000000
- current_score: -111.070097
- gap_to_target: 311.070097
- target_achievement_ratio: -55.535%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -111.070097）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    观测:
    obs[0]: x_position_relative_to_target
    obs[1]: y_position_relative_to_pad_height
    obs[2]: x_velocity
    obs[3]: y_velocity
    obs[4]: body_angle
    obs[5]: angular_velocity
    obs[6]: left_contact_flag (0.0/1.0)
    obs[7]: right_contact_flag (0.0/1.0)
    """

    def distance(obs_arr):
        return (obs_arr[0] ** 2 + obs_arr[1] ** 2 + 1e-8) ** 0.5

    # ---- 1. Progress reward: 距离减少量 ----
    dist_old = distance(obs)
    dist_new = distance(next_obs)
    progress_reward = dist_old - dist_new

    # ---- 2. Speed tracking reward ----
    max_speed = 5.0
    d_ref = 1.0
    desired_speed = max_speed * min(dist_new / d_ref, 1.0)

    cur_speed = (next_obs[2] ** 2 + next_obs[3] ** 2 + 1e-8) ** 0.5
    speed_error = abs(cur_speed - desired_speed)
    lambda_speed = 0.2
    speed_tracking_reward = -lambda_speed * speed_error

    # ---- 3. Landing improvement: 着陆质量改善量 (state_to_improvement) ----
    proximity_threshold = 0.5
    speed_threshold = 0.25
    angle_threshold = 0.2

    def landing_quality(o):
        d = (o[0] ** 2 + o[1] ** 2 + 1e-8) ** 0.5
        s = (o[2] ** 2 + o[3] ** 2 + 1e-8) ** 0.5
        prox = max(0.0, 1.0 - d / proximity_threshold)
        slow = max(0.0, 1.0 - s / speed_threshold)
        ang = max(0.0, 1.0 - abs(o[4]) / angle_threshold)
        contact = (o[6] + o[7]) * 0.5
        return prox + slow + ang + 2.0 * contact

    Q_old = landing_quality(obs)
    Q_new = landing_quality(next_obs)

    landing_improvement_coeff = 5.0
    landing_improvement = landing_improvement_coeff * max(0.0, Q_new - Q_old)

    # ---- 总奖励 ----
    total_reward = progress_reward + speed_tracking_reward + landing_improvement

    components = {
        'progress_reward': progress_reward,
        'speed_tracking_reward': speed_tracking_reward,
        'landing_improvement': landing_improvement
    }
    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    假设环境观测结构:
    obs[0]: x_position_relative_to_target
    obs[1]: y_position_relative_to_pad_height
    obs[2]: x_velocity
    obs[3]: y_velocity
    obs[4]: body_angle
    obs[5]: angular_velocity
    obs[6]: left_contact_flag (0.0/1.0)
    obs[7]: right_contact_flag (0.0/1.0)
    """
    # ---- 1. Progress reward: 距离减少量 (密集主学习信号) ----
    def distance(obs_arr):
        return (obs_arr[0]**2 + obs_arr[1]**2 + 1e-8) ** 0.5

    dist_old = distance(obs)
    dist_new = distance(next_obs)
    progress_reward = dist_old - dist_new

    # ---- 2. Speed tracking reward: 期望速度引导，代替硬惩罚 ----
    # 期望速度大小与距离成线性（远快近慢），用于引导合理的飞行速度
    max_speed = 5.0          # 预设的最大期望速度
    d_ref = 1.0              # 参考距离，在此距离下期望速度达到 max_speed
    desired_speed = max_speed * min(dist_new / d_ref, 1.0)

    cur_speed = (next_obs[2]**2 + next_obs[3]**2 + 1e-8) ** 0.5
    speed_error = abs(cur_speed - desired_speed)
    lambda_speed = 0.2
    speed_tracking_reward = -lambda_speed * speed_error

    # ---- 3. Soft landing proxy: 多条件线性组合，避免乘积梯度塌缩 ----
    proximity_threshold = 0.5
    speed_threshold = 0.25
    angle_threshold = 0.2     # 弧度

    proximity_score = max(0.0, 1.0 - dist_new / proximity_threshold)
    speed_low_score = max(0.0, 1.0 - cur_speed / speed_threshold)
    angle_score = max(0.0, 1.0 - abs(next_obs[4]) / angle_threshold)
    contact_score = (next_obs[6] + next_obs[7]) * 0.5

    # 线性组合，接触条件权重略高
    soft_landing_proxy = (
        proximity_score +
        speed_low_score +
        angle_score +
        2.0 * contact_score
    )  # 最大可能值约为 5.0

    # ---- 总奖励 ----
    total_reward = progress_reward + speed_tracking_reward + soft_landing_proxy

    components = {
        'progress_reward': progress_reward,
        'speed_tracking_reward': speed_tracking_reward,
        'soft_landing_proxy': soft_landing_proxy
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-111.070097, len=69.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-129.255058, -90.576126]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_tracking_reward | -43.020149 | -65.3% | 65.3% | 100.0% |
| landing_improvement | 21.697975 | 32.9% | 32.9% | 47.8% |
| progress_reward | 1.116579 | 1.7% | 1.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 飞行器着陆任务。飞行器从视口上部中心附近出发，受随机初始力影响。目标是**尽快平稳地降落在中央的着陆垫上**，同时**尽可能少地使用引擎推力**。智能体需要学习：向目标靠近、减速、保持机身姿态稳定、实现安全触垫。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x位置（相对着陆垫的水平坐标）
- obs[1]: y位置（相对垫高度的垂直坐标）
- obs[2]: x方向线速度
- obs[3]: y方向线速度
- obs[4]: 机身角度（俯仰角）
- obs[5]: 角速度
- obs[6]: 左支撑接触标志（1.0 接触，0.0 未接触）
- obs[7]: 右支撑接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: 无引擎（不施加推力）
- action 1: 左侧姿控引擎点火
- action 2: 主引擎点火（向下推力）
- action 3: 右侧姿控引擎点火

## 5. step 与终止条件分析
### 5.1 终止模式
Termination 由以下三种条件触发（任一为真即结束）：
1. `crash_or_body_contact`：坠毁或身体与地面/其他物体发生接触（大概率是失败）。
2. `horizontal_position_outside_viewport`：水平位置超出边界（失败）。
3. `body_not_awake_or_settled`：机体进入休眠/稳定状态。当飞行器完全静止、不再受物理唤醒时触发。若此时已处于着陆垫上且支撑腿接触，则通常表示成功着陆；若在坠毁后静止，则被前面的 crash 条件先行触发。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（step 返回的 info 为空，无显式标记）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典 `{}`，不可使用任何字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用。由于源码显式返回 `{}`，不能假设存在 `success`、`termination_reason` 等键。

## 7. 可用于奖励函数的信号
- **位置**：相对着陆垫的 x, y 坐标（obs[0], obs[1]），可直接用于评估靠近/悬停/触垫。
- **速度**：x, y 线速度（obs[2], obs[3]），可用于惩罚高速撞击或奖励稳定减速。
- **姿态**：机身角度（obs[4]），角速度（obs[5]），用于鼓励保持直立姿态。
- **接触**：左右支撑接触标志（obs[6], obs[7]），可用于检测是否着陆成功、是否平稳（双脚着垫）。
- **动作/引擎使用**：动作编号本身，可以用于惩罚推力使用（动作 1,2,3 为引擎点火）以鼓励省油。

# Compact expert route context
- selected_route_id: navigation_goal_reaching
- recommended_design_roles: terminal_success_reward (终点成功奖励), terminal_failure_penalty (失败惩罚), time_penalty (每步时间惩罚), distance_reward (距离型密集奖励), progress_delta_reward (增量进步奖励), potential_based_shaping (势能塑形奖励), gated_reward (门控/层级奖励)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -108.32 | -108.32 | 0.00 | 68.50 | progress_reward=0.033 soft_landing_bonus=0.005 stability_penalty=-0.136 | new_best |
| 2 | progress_reward + soft_landing_bonus + stability_penalty | -106.27 | -106.27 | 0.00 | 68.55 | progress_reward=0.033 soft_landing_bonus=0.005 stability_penalty=-0.014 | new_best |
| 3 | progress_reward + soft_landing_bonus + stability_penalty | -107.99 | -106.27 | -1.72 | 68.50 | progress_reward=0.033 soft_landing_bonus=0.009 stability_penalty=-0.014 | no_meaningful_improvement |
| 4 | progress_reward + soft_landing_bonus + stability_penalty | -92.00 | -92.00 | 0.00 | 68.95 | progress_reward=0.033 soft_landing_bonus=0.012 stability_penalty=-0.014 | same_skeleton_persistent_negative_fresh_restart |
| 5 | progress_reward + soft_landing_proxy + speed_tracking_reward | 162.94 | 162.94 | 0.00 | 463.70 | progress_reward=0.004 soft_landing_proxy=2.229 speed_tracking_reward=-0.374 | new_best |
| 6 | landing_improvement + progress_reward + speed_tracking_reward | -111.07 | 162.94 | -274.01 | 69.20 | landing_improvement=0.278 progress_reward=0.016 speed_tracking_reward=-0.620 | no_meaningful_improvement |
