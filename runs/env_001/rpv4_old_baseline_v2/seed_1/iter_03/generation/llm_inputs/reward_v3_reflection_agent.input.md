# 1. Search objective
- target_score: 200.000000
- current_score: -60.941087
- gap_to_target: 260.941087

# 2. Current reward program (score: -60.941087)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    x, y, vx, vy, angle, ang_vel = next_obs[0], next_obs[1], next_obs[2], next_obs[3], next_obs[4], next_obs[5]
    left, right = next_obs[6], next_obs[7]

    # 1. Progress toward the target centre (0,0)
    prev_x, prev_y = obs[0], obs[1]
    prev_dist = (prev_x ** 2 + prev_y ** 2) ** 0.5
    curr_dist = (x ** 2 + y ** 2) ** 0.5
    progress = prev_dist - curr_dist
    progress_reward = 2.0 * progress

    # 2. Attitude penalty (keep upright)
    angle_pen = -0.5 * (angle ** 2)

    # 3. Angular velocity penalty (smooth rotation)
    angvel_pen = -0.1 * (ang_vel ** 2)

    # 4. Linear velocity penalty (gentle movement)
    vel_pen = -0.2 * (vx ** 2 + vy ** 2)

    # 5. Contact reward (any leg touch + simultaneous bonus)
    contact_reward = 1.0 * left + 1.0 * right + 3.0 * left * right

    total = progress_reward + angle_pen + angvel_pen + vel_pen + contact_reward
    components = {
        "progress": progress_reward,
        "angle_penalty": angle_pen,
        "angvel_penalty": angvel_pen,
        "vel_penalty": vel_pen,
        "contact_reward": contact_reward
    }
    return float(total), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-60.941087, len=975.750000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-102.791515, -15.406515]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_penalty | -5.955435 | -44.6% | 44.6% | 100.0% |
| progress | 1.471255 | 11.0% | 21.0% | 100.0% |
| angle_penalty | -2.304841 | -17.3% | 17.3% | 100.0% |
| contact_reward | 2.050000 | 15.4% | 15.4% | 0.1% |
| angvel_penalty | -0.245465 | -1.8% | 1.8% | 100.0% |

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