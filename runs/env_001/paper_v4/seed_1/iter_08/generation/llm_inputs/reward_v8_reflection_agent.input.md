# Duplicate reward retry
The previous generation duplicated iter 7 (runs\env_001\paper_v4\seed_1\iter_07\generation\reward_v7.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 主学习信号：距离改善 × 姿态门控（保持不变）
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next

    angle_abs = abs(next_obs[4])
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)
    shaped_progress = 5.0 * progress_raw * gate

    # 2. 安全约束：角速度轻量惩罚（保持不变）
    ang_vel_penalty = -0.1 * (next_obs[5] ** 2)

    # 3. 着陆奖励：软转移事件（替代持续状态奖励）
    # 检测腿部接触的增量：接触增加即给分，完全接触后增量归零自然防刷
    contact_now = next_obs[6] * next_obs[7]
    contact_before = obs[6] * obs[7]
    new_contact = contact_now - contact_before
    if new_contact < 0.0:
        new_contact = 0.0

    # 着陆质量因子：靠近目标、直立、低速
    near_target = 1.0 / (1.0 + 3.0 * abs(next_obs[0]))
    upright = 1.0 / (1.0 + 3.0 * angle_abs)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    low_speed = 1.0 / (1.0 + speed)

    landing_bonus = 15.0 * new_contact * near_target * upright * low_speed

    total_reward = shaped_progress + ang_vel_penalty + landing_bonus
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus
    }
    return (float(total_reward), components)
```

# Search objective
- target_score: 200.000000
- current_score: 24.338936
- gap_to_target: 175.661064
- target_achievement_ratio: 12.169%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 24.338936）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 主学习信号：距离改善 × 姿态门控（保持不变）
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next

    angle_abs = abs(next_obs[4])
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)
    shaped_progress = 5.0 * progress_raw * gate

    # 2. 安全约束：角速度轻量惩罚（保持不变）
    ang_vel_penalty = -0.1 * (next_obs[5] ** 2)

    # 3. 着陆奖励：软转移事件（替代持续状态奖励）
    # 检测腿部接触的增量：接触增加即给分，完全接触后增量归零自然防刷
    contact_now = next_obs[6] * next_obs[7]
    contact_before = obs[6] * obs[7]
    new_contact = contact_now - contact_before
    if new_contact < 0.0:
        new_contact = 0.0

    # 着陆质量因子：靠近目标、直立、低速
    near_target = 1.0 / (1.0 + 3.0 * abs(next_obs[0]))
    upright = 1.0 / (1.0 + 3.0 * angle_abs)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    low_speed = 1.0 / (1.0 + speed)

    landing_bonus = 15.0 * new_contact * near_target * upright * low_speed

    total_reward = shaped_progress + ang_vel_penalty + landing_bonus
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus
    }
    return (float(total_reward), components)
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=24.338936, len=575.850000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-492.143659, 247.188147]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 390.287154 | 98.1% | 98.1% | 6.3% |
| shaped_progress | 3.494042 | 0.9% | 1.3% | 99.2% |
| angular_vel_penalty | -2.144577 | -0.5% | 0.5% | 97.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
该环境是一个 2D 飞行器轨迹优化任务。智能体需要从初始位置（视口顶部中心附近）出发，快速、稳定地降落到画面中央的目标接触垫上，同时尽可能少使用引擎推力。任务要求智能体学会靠近目标区域、减速、保持机体垂直，并用两条着陆腿安全接触垫面。

## 3. 观察空间 observation_space
-

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angular_vel_penalty + landing_bonus + shaped_progress | -42.74 | -42.74 | 0.00 | 975.65 | angular_vel_penalty=-0.005 landing_bonus=3.387 shaped_progress=0.003 | new_best |
| 2 | angular_vel_penalty + landing_bonus + shaped_progress | -17.88 | -17.88 | 0.00 | 759.35 | angular_vel_penalty=-0.004 landing_bonus=3.324 shaped_progress=0.003 | new_best |
| 3 | angular_vel_penalty + landing_bonus + shaped_progress | 27.72 | 27.72 | 0.00 | 495.55 | angular_vel_penalty=-0.005 landing_bonus=2.663 shaped_progress=0.138 | new_best |
| 4 | angular_vel_penalty + landing_bonus + shaped_progress | -29.48 | 27.72 | -57.19 | 76.95 | angular_vel_penalty=-0.017 landing_bonus=0.259 shaped_progress=0.347 | no_meaningful_improvement |
| 5 | angular_vel_penalty + landing_bonus + shaped_progress | 120.02 | 120.02 | 0.00 | 596.90 | angular_vel_penalty=-0.007 landing_bonus=3.361 shaped_progress=0.026 | new_best |
| 6 | angular_vel_penalty + landing_quality + shaped_progress | -16.87 | 120.02 | -136.89 | 1000.00 | angular_vel_penalty=-0.003 landing_quality=2.719 shaped_progress=0.012 | no_meaningful_improvement |
| 7 | angular_vel_penalty + landing_bonus + shaped_progress | 24.34 | 120.02 | -95.68 | 575.85 | angular_vel_penalty=-0.008 landing_bonus=0.307 shaped_progress=0.032 | no_meaningful_improvement |
