# 1. Search objective
- target_score: 200.000000
- current_score: -46.323659
- gap_to_target: 246.323659

# 2. Current reward program (score: -46.323659)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    px, py = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_px, prev_py = obs[0], obs[1]

    curr_dist = (px**2 + py**2)**0.5
    prev_dist = (prev_px**2 + prev_py**2)**0.5

    # 主正向驱动：向中心移动的进步奖励
    progress = 5.0 * (prev_dist - curr_dist)

    # 接近目标的正奖励，替代原来的距离惩罚
    proximity = 0.5 / (1.0 + curr_dist)

    # 速度惩罚（降低系数，允许探索加速）
    vel_penalty = -0.1 * (vx**2 + vy**2)

    # 保持直立
    angle_penalty = -0.5 * (angle**2)

    # 抑制过度旋转
    angvel_penalty = -0.02 * (ang_vel**2)

    # 接触奖励（降低幅度，避免落地后数值占主导）
    contact_reward = 0.5 * left_contact + 0.5 * right_contact + 1.0 * (left_contact * right_contact)

    # 稀疏成功奖励，大幅提高以凸显着陆目标
    success_bonus = 10.0 if (left_contact and right_contact and curr_dist < 0.5) else 0.0

    total = (progress + proximity + vel_penalty + angle_penalty +
             angvel_penalty + contact_reward + success_bonus)

    components = {
        "progress": progress,
        "proximity": proximity,
        "vel_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus
    }
    return float(total), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-46.323659, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-77.088527, -8.231464]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity | 307.998292 | 97.2% | 97.2% | 100.0% |
| progress | 4.670757 | 1.5% | 1.7% | 100.0% |
| vel_penalty | -1.862840 | -0.6% | 0.6% | 100.0% |
| angle_penalty | -1.468444 | -0.5% | 0.5% | 100.0% |
| angvel_penalty | -0.031268 | -0.0% | 0.0% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 3.5. Checkpoint evaluations (every 10% of training)
| pct | score_mean | score_std |
|---|---|---|
| 40% | 0.00 | 0.00 |
| 80% | 0.00 | 0.00 |

# 4. Environment facts
## 1. 任务目标
该环境是一个 2D 飞行器轨迹优化任务。智能体需要从初始位置（视口顶部中心附近）出发，快速、稳定地降落到画面中央的目标接触垫上，同时尽可能少使用引擎推力。任务要求智能体学会靠近目标区域、减速、保持机体垂直，并用两条着陆腿安全接触垫面。

## 3. 观察空间 observation_space
-