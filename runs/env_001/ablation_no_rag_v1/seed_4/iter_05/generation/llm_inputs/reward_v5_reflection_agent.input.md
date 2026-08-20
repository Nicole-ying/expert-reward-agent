# Search objective
- target_score: 200.000000
- current_score: 238.204421
- gap_to_target: -38.204421
- target_achievement_ratio: 119.102%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 238.204421）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]          # 相对目标平台水平坐标
    y = next_obs[1]          # 相对平台高度
    vx = next_obs[2]         # 水平速度
    vy = next_obs[3]         # 垂直速度
    angle = next_obs[4]      # 机身角度
    omega = next_obs[5]      # 角速度
    left_contact = next_obs[6]   # 左支撑接触标志
    right_contact = next_obs[7]  # 右支撑接触标志

    w_x = 0.05
    w_y = 0.05
    w_vx = 0.05
    w_vy = 0.05
    w_angle = 0.05
    w_omega = 0.05
    w_landing = 1.0

    target_proximity = -w_x * abs(x) - w_y * abs(y)
    velocity_penalty = -w_vx * (vx ** 2) - w_vy * (vy ** 2)
    orientation_penalty = -w_angle * (angle ** 2)
    angvel_penalty = -w_omega * (omega ** 2)
    landing_reward = w_landing * (left_contact * right_contact)

    total_reward = target_proximity + velocity_penalty + orientation_penalty + angvel_penalty + landing_reward

    components = {
        "target_proximity": target_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_reward": landing_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=238.204421, len=397.700000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[140.758031, 271.867349]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 77.950000 | 87.7% | 87.7% | 19.6% |
| target_proximity | -8.682168 | -9.8% | 9.8% | 100.0% |
| velocity_penalty | -1.941557 | -2.2% | 2.2% | 97.1% |
| orientation_penalty | -0.224519 | -0.3% | 0.3% | 100.0% |
| angvel_penalty | -0.120384 | -0.1% | 0.1% | 94.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
一个二维飞行器从视口顶部中央附近出发（带有随机初始扰动），需要尽可能快地到达并稳定停靠在中央目标平台上，同时尽量减少发动机推力。智能体应当学会靠近目标、减速、保持稳定姿态并安全接触平台。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: float32（推断）  
- obs[0]: 相对于目标平台的水平坐标  
- obs[1]: 相对于平台高度的垂直坐标  
- obs[2]: 水平线速度  
- obs[3]: 垂直线速度  
- obs[4]: 机身朝向角度  
- obs[5]: 角速度  
- obs[6]: 左侧支撑接触标志（0.0或1.0）  
- obs[7]: 右侧支撑接触标志（0.0或1.0）

## 4. 动作空间 action_space
- type: Discrete (n=4)  
- action 0: no_engine —— 不启动任何引擎  
- action 1: left_orientation_engine —— 启动左姿态调节引擎（产生逆时针/右转力矩）  
- action 2: main_engine —— 启动主引擎（提供向上推力）  
- action 3: right_orientation_engine —— 启动右姿态调节引擎（产生顺时针/左转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机身静止或已稳定，推测代表成功着陆并稳定）  
- failure-like termination: crash_or_body_contact（碰撞/危险身体接触）、horizontal_position_outside_viewport（水平位置超出边界）  
- ambiguous termination: 无（但body_not_awake_or_settled是否为绝对成功需结合任务理解，通常视为成功）  
- truncation: 未在 step 源码中提供（不排除有最大步数限制，但未暴露，不可假设）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 固定为空字典 {}）  
- forbidden_or_uncertain_info_fields: 所有 info 字段（因为 info 为空，任何假设都不安全）

## 7. 可用于奖励函数的信号
- position: obs[0]（相对水平位置）、obs[1]（相对高度）  
- velocity: obs[2]（水平速度）、obs[3]（垂直速度）  
- orientation: obs[4]（角度）、obs[5]（角速度）  
- contact: obs[6]（左接触）、obs[7]（右接触）  
- action/engine: 通过 action 参数获知当前使用的引擎类型（是否开主引擎、姿态引擎等）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angvel_penalty + orientation_penalty + target_proximity + velocity_penalty | -53.29 | -53.29 | 0.00 | 72.70 | angvel_penalty=-0.011 orientation_penalty=-0.001 target_proximity=-0.212 velocity_penalty=-0.063 | new_best |
| 2 | angvel_penalty + orientation_penalty + target_proximity + velocity_penalty | -87.67 | -53.29 | -34.38 | 1000.00 | angvel_penalty=-0.005 orientation_penalty=-0.007 target_proximity=7.147 velocity_penalty=-0.021 | no_meaningful_improvement |
| 3 | angvel_penalty + landing_reward + orientation_penalty + target_proximity + velocity_penalty | 102.16 | 102.16 | 0.00 | 413.80 | angvel_penalty=-0.003 landing_reward=0.357 orientation_penalty=-0.001 target_proximity=-0.138 velocity_penalty=-0.029 | new_best |
| 4 | angvel_penalty + landing_reward + orientation_penalty + target_proximity + velocity_penalty | 238.20 | 238.20 | 0.00 | 397.70 | angvel_penalty=-0.002 landing_reward=0.550 orientation_penalty=-0.001 target_proximity=-0.024 velocity_penalty=-0.015 | target_solved_new_best |
