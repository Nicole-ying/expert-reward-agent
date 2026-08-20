# Search objective
- target_score: 200.000000
- current_score: -53.287992
- gap_to_target: 253.287992
- target_achievement_ratio: -26.644%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -53.287992）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 下一状态的关键信号
    x = next_obs[0]    # 相对目标平台的水平坐标
    y = next_obs[1]    # 相对于平台高度的垂直坐标
    vx = next_obs[2]   # 水平线速度
    vy = next_obs[3]   # 垂直线速度
    angle = next_obs[4] # 机身朝向角度
    omega = next_obs[5] # 角速度

    # 权重设置（可在后续迭代中调整）
    w_x = 0.2
    w_y = 0.2
    w_vx = 0.05
    w_vy = 0.05
    w_angle = 0.05
    w_omega = 0.05

    # 主学习信号：靠近目标（位置绝对值越小，惩罚越小）
    target_proximity = -w_x * abs(x) - w_y * abs(y)

    # 稳定/安全约束：抑制高速运动与机身摆动
    velocity_penalty = -w_vx * (vx ** 2) - w_vy * (vy ** 2)
    orientation_penalty = -w_angle * (angle ** 2)
    angvel_penalty = -w_omega * (omega ** 2)

    total_reward = target_proximity + velocity_penalty + orientation_penalty + angvel_penalty

    components = {
        "target_proximity": target_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "angvel_penalty": angvel_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-53.287992, len=72.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-88.484955, -17.388169]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| target_proximity | -15.139127 | -76.7% | 76.7% | 100.0% |
| velocity_penalty | -4.241642 | -21.5% | 21.5% | 99.8% |
| angvel_penalty | -0.330935 | -1.7% | 1.7% | 99.7% |
| orientation_penalty | -0.020351 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 11/20
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
