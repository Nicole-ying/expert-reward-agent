# 1. Search objective
- target_score: 200.000000
- current_score: 52.523801
- gap_to_target: 147.476199

# 2. Current reward program (score: 52.523801)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state
    px, py, vx, vy, angle, ang_vel = next_obs[0], next_obs[1], next_obs[2], next_obs[3], next_obs[4], next_obs[5]
    left_contact, right_contact = next_obs[6], next_obs[7]

    # ---- New core signal: distance penalty ----
    # Agent is heavily pushed to minimise its distance to the centre (0,0)
    curr_dist = (px ** 2 + py ** 2) ** 0.5
    distance_penalty = -0.5 * curr_dist

    # ---- Attitude (keep upright) ----
    angle_penalty = -0.5 * (angle ** 2)

    # ---- Rotation smoothness ----
    angvel_penalty = -0.1 * (ang_vel ** 2)

    # ---- Linear speed: now a gentle penalty (was 0.2, dwarfed everything) ----
    vel_penalty = -0.02 * (vx ** 2 + vy ** 2)

    # ---- Landing contact: much stronger incentive ----
    contact_reward = 10.0 * left_contact + 10.0 * right_contact + 30.0 * left_contact * right_contact

    total = distance_penalty + angle_penalty + angvel_penalty + vel_penalty + contact_reward
    components = {
        "distance_penalty": distance_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "vel_penalty": vel_penalty,
        "contact_reward": contact_reward
    }
    return float(total), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=52.523801, len=507.200000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-91.792933, 208.111232]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 6100.000000 | 97.8% | 97.8% | 29.5% |
| distance_penalty | -129.115543 | -2.1% | 2.1% | 100.0% |
| angle_penalty | -4.202416 | -0.1% | 0.1% | 100.0% |
| vel_penalty | -1.095749 | -0.0% | 0.0% | 99.9% |
| angvel_penalty | -0.511493 | -0.0% | 0.0% | 96.8% |

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