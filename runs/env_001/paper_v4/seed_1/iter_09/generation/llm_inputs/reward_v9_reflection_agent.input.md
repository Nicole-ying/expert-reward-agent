# Search objective
- target_score: 200.000000
- current_score: -31.051550
- gap_to_target: 231.051550
- target_achievement_ratio: -15.526%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -31.051550）
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

    # 3. 着陆奖励：转移事件门控 × 加性质量因子（product→weighted sum，消除塌缩）
    contact_now = next_obs[6] * next_obs[7]
    contact_before = obs[6] * obs[7]
    new_contact = contact_now - contact_before
    if new_contact < 0.0:
        new_contact = 0.0

    near_target = 1.0 / (1.0 + 3.0 * abs(next_obs[0]))
    upright = 1.0 / (1.0 + 3.0 * angle_abs)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    low_speed = 1.0 / (1.0 + speed)

    # 加性聚合：各质量因子独立贡献，避免乘积塌缩
    landing_bonus = new_contact * (2.0 * near_target + 0.5 * upright + 0.5 * low_speed)

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
score=-31.051550, len=76.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-72.228187, 12.175979]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| shaped_progress | 5.258439 | 68.4% | 71.7% | 99.9% |
| landing_bonus | 1.499009 | 19.5% | 19.5% | 1.0% |
| angular_vel_penalty | -0.674864 | -8.8% | 8.8% | 99.5% |

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
| 8 | angular_vel_penalty + landing_bonus + shaped_progress | -31.05 | 120.02 | -151.07 | 76.45 | angular_vel_penalty=-0.011 landing_bonus=0.019 shaped_progress=0.066 | unsolved_high_achievement_continue_from_best |
