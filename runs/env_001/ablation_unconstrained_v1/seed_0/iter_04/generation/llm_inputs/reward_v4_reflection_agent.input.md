# Search objective
- target_score: 200.000000
- current_score: 121.869068
- gap_to_target: 78.130932
- target_achievement_ratio: 60.935%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 121.869068）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Next state
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 1. Bounded proximity: always provides gradient toward target
    #    (0, 1], peaks at 1 when exactly on target
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    r_dist = 1.0 / (1.0 + distance)
    
    # 2. Stability penalty: discourage violent motion, promote gentle flight
    r_stability = -(
        0.01 * (abs(x_vel) + abs(y_vel)) +
        0.1 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )
    
    # 3. Continuous soft-landing proxy: product of four bounded factors [0,1]
    #    Each factor opens gradually — provides gradient before perfect conditions.
    proximity = max(0.0, 1.0 - distance / 0.5)          # activates when dist < 0.5
    stillness = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.4)  # activates when |v| < 0.4
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.3)  # activates when |angle| < 0.3
    contact   = (left_contact + right_contact) / 2.0    # 0, 0.5, or 1.0
    # Floor of 0.1 ensures weak signal before any leg contact;
    # one leg = 0.55, both legs = 1.0 — gradient toward full touchdown.
    r_landing = 8.0 * proximity * stillness * upright * (0.1 + 0.9 * contact)
    
    total_reward = r_dist + r_stability + r_landing
    
    components = {
        "distance_reward": r_dist,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }
    
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=121.869068, len=837.850000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[29.151419, 204.519208]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 689.224037 | 52.0% | 52.0% | 75.4% |
| distance_reward | 629.048129 | 47.5% | 47.5% | 100.0% |
| stability_penalty | -6.188048 | -0.5% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个2D飞行器轨迹优化任务。  
主体从视口顶部中央附近出发，并受到随机初始力。  
目标是**尽可能快地飞到并稳定降落在中央目标垫上**，同时**消耗尽可能少的引擎推力**。  
智能体需要学会：接近目标、减速、保持姿态稳定、实现安全触地。

## 3. 观察空间 observation_space
- type: Box（连续）
- shape: [8]
- dtype: float64 （默认，部分字段为0/1浮点）
- obs[0]: x_position —— 相对目标垫的水平坐标
- obs[1]: y_position —— 相对目标垫高度的垂直坐标
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体朝向角
- obs[5]: angular_velocity —— 角速度
- obs[6]: left_support_contact —— 左支撑腿接触标志（1.0 接触; 0.0 未接触）
- obs[7]: right_support_contact —— 右支撑腿接触标志（1.0 接触; 0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine —— 不点燃任何引擎
- action 1: left_orientation_engine —— 点燃左侧姿态引擎
- action 2: main_engine —— 点燃主引擎（产生推力）
- action 3: right_orientation_engine —— 点燃右侧姿态引擎（与左相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled —— 机体不再运动且稳定，可能对应成功着陆
- failure-like termination: crash_or_body_contact —— 机体重重撞击或非支撑部位触地（并非正常支撑腿触地）
- failure-like termination: horizontal_position_outside_viewport —— 飞出水平边界
- ambiguous termination: 无
- truncation: 源码中未展示步数限制，但通常环境存在episode截断，这里无信息，不依赖

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典）
- explicit_failure_flag_available: false
- allowed_info_fields: （无）
- forbidden_or_uncertain_info_fields: info.* （任何 info 字段都未声明，不可安全使用）

## 7. 可用于奖励函数的信号
- **位置**：obs[0] x_position, obs[1] y_position（相对目标垫的水平和垂直距离）
- **速度**：obs[2] x_velocity, obs[3] y_velocity（下降速度等）
- **姿态与角速度**：obs[4] body_angle, obs[5] angular_velocity
- **接触**：obs[6] left_support_contact, obs[7] right_support_contact（是否两条支撑腿着地）
- **动作/引擎**：当前动作类型（是否使用主引擎或姿态引擎）

# Compact expert route context
- selected_route_id: navigation_goal_reaching
- recommended_design_roles: terminal_success_reward (终点成功奖励), terminal_failure_penalty (失败惩罚), time_penalty (每步时间惩罚), distance_reward (距离型密集奖励), progress_delta_reward (增量进步奖励), potential_based_shaping (势能塑形奖励), gated_reward (门控/层级奖励)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.