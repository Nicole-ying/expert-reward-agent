# 1. Search objective
- target_score: 200.000000
- current_score: -47.425882
- gap_to_target: 247.425882

# 2. Current reward program (score: -47.425882)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    px, py = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_left = obs[6]
    prev_right = obs[7]

    curr_dist = (px**2 + py**2)**0.5

    # Core drive toward origin (proven in iter 3, ~-130/ep)
    distance_penalty = -0.5 * curr_dist

    # One-time contact onset bonus — fires only on transition 0→1
    left_onset = 1.0 if (left_contact > 0.5 and prev_left < 0.5) else 0.0
    right_onset = 1.0 if (right_contact > 0.5 and prev_right < 0.5) else 0.0
    contact_reward = 40.0 * left_onset + 40.0 * right_onset + 80.0 * (left_onset * right_onset)

    # Per-step stable-landing bonus (small, only during successful touchdown)
    success_bonus = 2.0 if (left_contact and right_contact and curr_dist < 0.3 and abs(angle) < 0.2) else 0.0

    # Keep upright
    angle_penalty = -0.5 * (angle ** 2)

    # Moderate speed penalty
    vel_penalty = -0.1 * (vx ** 2 + vy ** 2)

    # Smooth rotation
    angvel_penalty = -0.05 * (ang_vel ** 2)

    total = (distance_penalty + contact_reward + success_bonus +
             angle_penalty + vel_penalty + angvel_penalty)

    components = {
        "distance_penalty": distance_penalty,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "vel_penalty": vel_penalty,
        "angvel_penalty": angvel_penalty
    }
    return float(total), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-47.425882, len=75.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-93.804292, -4.433477]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 76.000000 | 62.8% | 62.8% | 2.1% |
| distance_penalty | -34.506254 | -28.5% | 28.5% | 100.0% |
| vel_penalty | -8.282890 | -6.8% | 6.8% | 100.0% |
| success_bonus | 1.400000 | 1.2% | 1.2% | 0.9% |
| angvel_penalty | -0.518872 | -0.4% | 0.4% | 99.9% |
| angle_penalty | -0.323311 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 8/20
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