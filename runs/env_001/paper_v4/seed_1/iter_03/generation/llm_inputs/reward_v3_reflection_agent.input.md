# Search objective
- target_score: 200.000000
- current_score: -17.878810
- gap_to_target: 217.878810
- target_achievement_ratio: -8.939%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -17.878810）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 匿名环境观测假设（经典2D着陆器）：
    # obs[0]: x (水平偏移)    obs[1]: y (高度)
    # obs[2]: vx             obs[3]: vy
    # obs[4]: angle          obs[5]: ang_vel
    # obs[6]: left_contact   obs[7]: right_contact

    # 1. 主学习信号：距离改善 × 姿态门控
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next          # 正值表示靠近目标

    angle_abs = abs(next_obs[4])                     # 机体角度（弧度）
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)        # soft health gate
    shaped_progress = progress_raw * gate

    # 2. 安全约束：角速度轻量惩罚
    ang_vel         = next_obs[5]
    ang_vel_penalty = -0.1 * (ang_vel ** 2)

    # 3. 着陆质量奖励：低速+姿态良好才给显著奖励
    left  = next_obs[6]
    right = next_obs[7]
    both_contact = left * right                      # 仅当两腿同时接触时非零

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    velocity_gate = 1.0 / (1.0 + 2.0 * speed)        # 低速→1，高速→0

    angle_gate = 1.0 / (1.0 + 5.0 * angle_abs)       # 直立→1，倾斜→0

    landing_bonus = 10.0 * both_contact * velocity_gate * angle_gate

    total_reward = shaped_progress + ang_vel_penalty + landing_bonus
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-17.878810, len=759.350000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-181.708159, 222.063453]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 49.682843 | 97.3% | 97.3% | 1.2% |
| shaped_progress | 0.829294 | 1.6% | 2.1% | 99.8% |
| angular_vel_penalty | -0.330852 | -0.6% | 0.6% | 99.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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
