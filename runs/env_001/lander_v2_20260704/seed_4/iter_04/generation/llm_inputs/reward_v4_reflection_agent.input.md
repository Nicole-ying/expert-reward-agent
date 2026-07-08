# 上一轮奖励函数代码（该轮得分: 236.458690）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 距离进展：每一步靠近目标的欧氏距离变化
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = 4.0 * (dist_obs - dist_next)

    # 稳定性约束：抑制过度速度和姿态变化
    stability_penalty = (
        -0.01 * (abs(next_obs[2]) + abs(next_obs[3]))
        - 0.01 * abs(next_obs[4])
        - 0.01 * abs(next_obs[5])
    )

    # 稠密着陆代理信号：连续评估各着陆条件的满足度
    D_x = 2.0
    D_y = 2.0
    D_v = 1.0
    D_angle = 0.5

    x_sat = max(0.0, 1.0 - abs(next_obs[0]) / D_x)
    y_sat = max(0.0, 1.0 - abs(next_obs[1]) / D_y)
    vx_sat = max(0.0, 1.0 - abs(next_obs[2]) / D_v)
    vy_sat = max(0.0, 1.0 - abs(next_obs[3]) / D_v)
    angle_sat = max(0.0, 1.0 - abs(next_obs[4]) / D_angle)
    contact_factor = max(0.05, 0.5 * (next_obs[6] + next_obs[7]))

    landing_proxy = x_sat * y_sat * vx_sat * vy_sat * angle_sat * contact_factor
    landing_reward = 0.8 * landing_proxy

    total_reward = progress_reward + stability_penalty + landing_reward

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_proxy': landing_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=236.458690, len=488.850000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[168.849399, 291.138404]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 215.579273 | 96.9% | 96.9% | 100.0% |
| progress_reward | 5.473114 | 2.5% | 2.5% | 98.0% |
| stability_penalty | -1.354567 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 飞行器轨迹优化任务。智能体从一个靠近视口顶部中心的位置出发，初始受到随机力扰动。  
核心目标：**尽快且稳定地降落在画面中央的目标平台上**（降低水平与垂直速度、保持姿态平稳、安全接触）。  
同时希望发动机推力使用越少越好（能耗经济性为附属优化项）。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（连续部分为浮点数，接触标志用 0.0/1.0 表示）
- obs[0] (x_position): 水平坐标，相对于目标平台中心的偏移量（朝右为正）
- obs[1] (y_position): 垂直坐标，相对于平台高度的偏移量（朝上为正）
- obs[2] (x_velocity): 水平线速度
- obs[3] (y_velocity): 垂直线速度
- obs[4] (body_angle): 机体朝向角度（单位：弧度）
- obs[5] (angular_velocity): 角速度
- obs[6] (left_support_contact): 左侧支撑接触标志，0.0 或 1.0
- obs[7] (right_support_contact): 右侧支撑接触标志，0.0 或 1.0

## 4. 动作空间 action_space
- type: Discrete(4)
- 各动作含义：
  - action 0: no_engine —— 不启用任何引擎（滑行/自由运动）
  - action 1: left_orientation_engine —— 点燃左侧姿态调整引擎（产生旋转力矩）
  - action 2: main_engine —— 点燃主引擎（主要推力，垂直于机体下方）
  - action 3: right_orientation_engine —— 点燃右侧（与左侧相反）姿态调整引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（机体不再活跃或已稳定，可能表示成功着陆并静止）
- failure-like termination: `crash_or_body_contact`（撞击或机体其他部位接触障碍）、`horizontal_position_outside_viewport`（水平出界）
- ambiguous termination: 无（所有终止条件均按上述归类，但需注意 `body_not_awake_or_settled` 未直接判断是否在目标位置，实际成功与否依赖于奖励设计）
- truncation: 未显式设置，理论上可能由时间限制截断（本环境未说明，可忽略）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回空字典 `{}`）
- forbidden_or_uncertain_info_fields: 任何 info 字段均不可用（因 info 为空）

## 7. 可用于奖励函数的信号
- position: `next_obs[0]`（水平偏移）、`next_obs[1]`（垂直偏移）
- velocity: `next_obs[2]`（水平速度）、`next_obs[3]`（垂直速度）
- orientation: `next_obs[4]`（机体角度）与 `next_obs[5]`（角速度）
- contact: `next_obs[6]` 和 `next_obs[7]`（左右支撑接触标志）
- action/engine: 可通过动作选择判断是否使用主引擎、姿态引擎，进而隐含能耗信号

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_bonus + progress_reward + stability_penalty | -94.43 | -94.43 | 0.00 | 68.70 | landing_bonus=0.000 progress_reward=0.016 stability_penalty=-0.014 | new_best |
| 2 | landing_proxy + progress_reward + stability_penalty | 171.00 | 171.00 | 0.00 | 526.55 | landing_proxy=0.859 progress_reward=0.003 stability_penalty=-0.004 | new_best |
| 3 | landing_proxy + progress_reward + stability_penalty | 236.46 | 236.46 | 0.00 | 488.85 | landing_proxy=0.407 progress_reward=0.012 stability_penalty=-0.004 | target_solved_new_best |
