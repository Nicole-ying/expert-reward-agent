# 1. Search objective
- target_score: 200.000000
- current_score: -80.353504
- gap_to_target: 280.353504

# 2. Current reward program (score: -80.353504)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next state
    px, py = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Unpack previous state for distance change
    prev_px, prev_py = obs[0], obs[1]

    curr_dist = (px**2 + py**2)**0.5
    prev_dist = (prev_px**2 + prev_py**2)**0.5

    # 1. Progress toward center
    dist_reward = 0.5 * (prev_dist - curr_dist)

    # 2. Small penalty for being far away (shapes final position)
    distance_penalty = -0.1 * curr_dist

    # 3. Keep upright
    angle_penalty = -0.5 * (angle**2)

    # 4. Encourage low speed
    vel_penalty = -0.2 * (vx**2 + vy**2)

    # 5. Penalize excessive rotation
    angvel_penalty = -0.02 * (ang_vel**2)

    # 6. Soft contact incentive (magnitude drastically reduced)
    contact_reward = 1.0 * left_contact + 1.0 * right_contact + 2.0 * (left_contact * right_contact)

    # 7. Success bonus for stable touchdown near center
    success_bonus = 2.0 if (left_contact and right_contact and curr_dist < 0.5) else 0.0

    total = (dist_reward + distance_penalty + angle_penalty +
             vel_penalty + angvel_penalty + contact_reward + success_bonus)

    components = {
        "dist_reward": dist_reward,
        "distance_penalty": distance_penalty,
        "angle_penalty": angle_penalty,
        "vel_penalty": vel_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus
    }
    return float(total), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-80.353504, len=813.750000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-222.921763, 195.269367]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 53.250000 | 52.0% | 52.0% | 1.8% |
| distance_penalty | -38.098124 | -37.2% | 37.2% | 100.0% |
| vel_penalty | -8.603585 | -8.4% | 8.4% | 100.0% |
| angle_penalty | -1.473811 | -1.4% | 1.4% | 100.0% |
| dist_reward | 0.378018 | 0.4% | 0.7% | 100.0% |
| success_bonus | 0.200000 | 0.2% | 0.2% | 0.0% |
| angvel_penalty | -0.031442 | -0.0% | 0.0% | 100.0% |

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