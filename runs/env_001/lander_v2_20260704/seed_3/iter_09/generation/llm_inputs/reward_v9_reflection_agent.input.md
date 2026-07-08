# 上一轮奖励函数代码（该轮得分: -7.385196）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. Main learning signal: progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = d_prev - d_next

    # 2. Stability constraint: light penalty on large speeds, tilt, and angular velocity.
    speed = abs(next_obs[2]) + abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    w_speed = 0.001
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = -w_speed * speed - w_angle * angle - w_angvel * ang_vel

    # 3. Soft landing proxy: continuous contact signal.
    contact_avg = (next_obs[6] + next_obs[7]) / 2.0
    dist = d_next
    k_dist = 2.0
    k_speed = 1.0
    k_angle = 5.0
    w_proxy = 5.0
    soft_landing_proxy = (w_proxy * contact_avg *
                          (2.718281828 ** (-k_dist * dist)) *
                          (2.718281828 ** (-k_speed * speed)) *
                          (2.718281828 ** (-k_angle * angle)))

    # 4. Engine efficiency penalty: discourage unnecessary engine use.
    engine_used = 0.0 if action == 0 else 1.0
    w_engine = 0.005
    engine_penalty = -w_engine * engine_used

    total_reward = approach_reward + stability_penalty + soft_landing_proxy + engine_penalty

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "engine_penalty": engine_penalty
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. Main learning signal: progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = d_prev - d_next

    # 2. Stability constraint: light penalty on large speeds, tilt, and angular velocity.
    speed = abs(next_obs[2]) + abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    w_speed = 0.001
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = -w_speed * speed - w_angle * angle - w_angvel * ang_vel

    # 3. Soft landing proxy: continuous contact signal replaces binary product.
    # (left + right)/2 gives 0/0.5/1.0 gradient for partial landings.
    contact_avg = (next_obs[6] + next_obs[7]) / 2.0
    dist = d_next
    k_dist = 2.0
    k_speed = 1.0
    k_angle = 5.0
    w_proxy = 5.0
    soft_landing_proxy = (w_proxy * contact_avg *
                          (2.718281828 ** (-k_dist * dist)) *
                          (2.718281828 ** (-k_speed * speed)) *
                          (2.718281828 ** (-k_angle * angle)))

    total_reward = approach_reward + stability_penalty + soft_landing_proxy

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-7.385196, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-38.415243, 26.868958]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| engine_penalty | -4.890000 | -69.3% | 69.3% | 97.8% |
| approach_reward | 1.319580 | 18.7% | 22.2% | 100.0% |
| stability_penalty | -0.597014 | -8.5% | 8.5% | 100.0% |
| soft_landing_proxy | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器从起点（视口顶部中心附近，有随机初始受力）移动并平稳降落到画面中央的目标平台。  
要求尽可能**快**地到达目标，同时**尽量少使用引擎推力**，并在接触时保持**姿态稳定**和**安全接触**。  
核心目标：到达目标平台并稳定停驻。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64（默认连续值）
- obs[0]：x_position —— 相对于目标平台的水平坐标  
- obs[1]：y_position —— 相对于平台高度的垂直坐标  
- obs[2]：x_velocity —— 水平线速度  
- obs[3]：y_velocity —— 垂直线速度  
- obs[4]：body_angle —— 机体倾角  
- obs[5]：angular_velocity —— 角速度  
- obs[6]：left_support_contact —— 左脚接触标志（1.0/0.0）  
- obs[7]：right_support_contact —— 右脚接触标志（1.0/0.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0：no_engine —— 无推力
- action 1：left_orientation_engine —— 启动左侧姿态引擎（产生角速度/微小推力）
- action 2：main_engine —— 启动主引擎（产生主要上升/减速推力）
- action 3：right_orientation_engine —— 启动右侧姿态引擎（与左边方向相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` —— 机体停止运动并稳定（即成功着陆并停驻）
- failure-like termination: 
  - `crash_or_body_contact` —— 任何非法的碰撞或身体接触（如撞到非平台区域、过猛撞击）
  - `horizontal_position_outside_viewport` —— 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 未出现在 step 中（`truncated` 返回 `False`），说明无时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（`info` 为空字典 `{}`）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用

## 7. 可用于奖励函数的信号
- position: next_obs[0] (水平位置误差), next_obs[1] (垂直位置/接近平台高度)  
- velocity: next_obs[2] (水平速度), next_obs[3] (垂直速度) —— 可用于惩罚硬着陆  
- orientation: next_obs[4] (倾角) —— 可用于要求平稳姿态  
- contact: next_obs[6], next_obs[7] (左右接触标志) —— 可奖励双脚稳定着地  
- action/engine: action id —— 可惩罚使用引擎推力以促进节能

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | approach_reward + soft_landing_proxy + stability_penalty | -95.25 | -95.25 | 0.00 | 68.80 | approach_reward=0.016 soft_landing_proxy=0.025 stability_penalty=-0.019 | new_best |
| 2 | approach_reward + soft_landing_proxy + stability_penalty | 149.44 | 149.44 | 0.00 | 564.95 | approach_reward=0.005 soft_landing_proxy=1.728 stability_penalty=-0.002 | new_best |
| 3 | approach_reward + soft_landing_proxy + stability_penalty | -10.51 | 149.44 | -159.95 | 1000.00 | approach_reward=0.088 soft_landing_proxy=1.673 stability_penalty=-0.002 | no_meaningful_improvement |
| 4 | approach_reward + soft_landing_proxy + stability_penalty | 136.73 | 149.44 | -12.71 | 237.45 | approach_reward=0.027 soft_landing_proxy=0.690 stability_penalty=-0.002 | same_skeleton_oscillation_fresh_restart |
| 5 | landing_reward + shaping_reward | -236.15 | 149.44 | -385.59 | 835.50 | landing_reward=0.002 shaping_reward=0.951 | no_meaningful_improvement |
| 6 | landing_reward + shaping_reward | -236.15 | 149.44 | -385.59 | 835.50 | landing_reward=0.002 shaping_reward=0.951 | no_meaningful_improvement |
| 7 | approach_reward + soft_landing_proxy + stability_penalty | 198.14 | 198.14 | 0.00 | 406.05 | approach_reward=0.003 soft_landing_proxy=1.691 stability_penalty=-0.002 | new_best |
| 8 | approach_reward + engine_penalty + soft_landing_proxy + stability_penalty | -7.39 | 198.14 | -205.53 | 1000.00 | approach_reward=0.004 engine_penalty=-0.004 soft_landing_proxy=1.661 stability_penalty=-0.002 | no_meaningful_improvement |
