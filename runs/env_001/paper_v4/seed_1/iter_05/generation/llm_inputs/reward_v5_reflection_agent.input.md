# Search objective
- target_score: 200.000000
- current_score: -29.475926
- gap_to_target: 229.475926
- target_achievement_ratio: -14.738%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -29.475926）
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
    shaped_progress = 25.0 * progress_raw * gate

    # 2. 安全约束：角速度轻量惩罚
    ang_vel_penalty = -0.1 * (next_obs[5] ** 2)

    # 3. 着陆奖励：转移事件 —— 仅首次触地瞬间触发
    prev_both = obs[6] * obs[7]                       # 上一步双腿接触
    curr_both = next_obs[6] * next_obs[7]             # 当前步双腿接触
    landing_transition = max(0.0, curr_both - prev_both)  # 0→1 上升沿

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    velocity_gate = 1.0 / (1.0 + 2.0 * speed)         # 低速→1，高速→0
    angle_gate = 1.0 / (1.0 + 5.0 * angle_abs)        # 直立→1，倾斜→0

    landing_bonus = 100.0 * landing_transition * velocity_gate * angle_gate

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
score=-29.475926, len=76.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-66.090146, 6.503554]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| shaped_progress | 24.534608 | 52.4% | 55.1% | 100.0% |
| landing_bonus | 20.359603 | 43.5% | 43.5% | 0.8% |
| angular_vel_penalty | -0.687755 | -1.5% | 1.5% | 99.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
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
