# Prompt Record

## System Prompt

```text
你是训练数据记者。你的唯一职责是：把训练反馈和迭代历史整理成客观的数据报告。你**不做诊断、不给建议、不推荐修改任何组件**。

# 输入数据

你有全部迭代的完整信息，不只是当前轮：

1. **本轮训练反馈** + **本轮 reward 代码**
2. **全部历史轮次的训练反馈 + reward 代码**（用于构建骨架演进表）
3. **历史记忆表**（每轮 skeleton、score、len、action）
4. **Checkpoint 评估数据**（每 10% 步数的官方得分轨迹）

# 输出格式

## 1. 行为事实
[用客观数字描述 agent 的行为，不做因果推断]
- 得分：score=X，range=[min, max]
- 终止方式：terminated X/20（成功/失败），truncated Y/20（时间上限）
- 回合长度：len=Z
- Checkpoint 轨迹：官方得分从 cp1=X1 → cp10=X10（上升/下降/持平）

## 2. 骨架演进表
```
| iter | score | len | term | compA_ep_sum | compA_active | compB_ep_sum | compB_active | ... |
| N-2  |       |     |      |              |              |              |              |     |
| N-1  |       |     |      |              |              |              |              |     |
| N    |       |     |      |              |              |              |              |     |
```

## 3. 数字标记
仅标记以下客观事实，不解释原因，不给建议：
- **主导**：某组件 magnitude_share > 80%
- **死组件**：某组件 active_rate < 2%
- **淹没**：某组件 active_rate > 90% 但 magnitude_share < 3%
- **全覆盖惩罚**：某组件 active_rate > 95% 且 signed_share 为负
- **量级对比**：列出每步均值（ep_sum ÷ len）最大的两个组件和最小的两个组件
- **修改变化**：如果存在上一轮，列出每个组件 ep_sum 和 active_rate 的 Δ

```

## User Prompt

```markdown
# 本轮训练反馈 (score=227.061788)
# Training Feedback

## Final-policy outcome
score=227.061788, len=490.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[190.921549, 273.107479]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_reward | 33.088806 | 83.9% | 83.9% | 7.0% |
| orientation_stability_reward | -3.035867 | -7.7% | 7.7% | 100.0% |
| progress_reward | 1.987577 | 5.0% | 5.2% | 97.1% |
| velocity_damping_reward | -1.277038 | -3.2% | 3.2% | 97.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估数据（每10%步数的官方环境得分）
| pct | score_mean | score_std |
|---|---|---|

# 本轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    #   Potential = -(x^2 + y^2), reward = potential_curr - potential_next
    #   Positive when distance to pad decreases, zero when stationary.
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 1.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: velocity_damping (proximity_gated quadratic penalty)
    #   Penalise speed more strongly when near the pad.
    # ------------------------------------------------------------
    dist_next_sqrt = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    gate = 1.0 / (1.0 + 2.0 * dist_next_sqrt)          # alpha = 2.0
    vel_penalty = 0.1 * (next_obs[2] ** 2 + next_obs[3] ** 2) * gate
    velocity_damping_reward = -vel_penalty

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    #   Keep body upright and angular velocity small.
    # ------------------------------------------------------------
    angle_penalty = 1.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.2 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: safe_contact_reward (joint_condition_proxy)
    #   Give positive reward when both legs touch the pad,
    #   vehicle is at pad height, and vertical velocity is low.
    #   (Coefficient increased from 0.5 to 1.0)
    # ------------------------------------------------------------
    both_contacts = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    # y position close to zero (pad height)
    proximity_to_pad = max(0.0, 1.0 - abs(next_obs[1]) / 0.2)   # threshold_y = 0.2
    # vertical velocity small
    soft_vel = max(0.0, 1.0 - abs(next_obs[3]) / 0.2)           # threshold_vy = 0.2
    safe_contact_reward = 1.0 * both_contacts * proximity_to_pad * soft_vel

    total_reward = (progress_reward +
                    velocity_damping_reward +
                    orientation_stability_reward +
                    safe_contact_reward)

    components = {
        "progress_reward": progress_reward,
        "velocity_damping_reward": velocity_damping_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "safe_contact_reward": safe_contact_reward,
    }

    return float(total_reward), components
```

# 历史 iter_01 训练反馈
# Training Feedback

## Final-policy outcome
score=144.172033, len=965.450000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[112.265517, 280.371299]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_reward | 312.464401 | 97.7% | 97.7% | 65.2% |
| orientation_stability_reward | -3.951835 | -1.2% | 1.2% | 100.0% |
| progress_reward | 1.985933 | 0.6% | 0.6% | 100.0% |
| velocity_damping_reward | -1.290861 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_01 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    #   Potential = -(x^2 + y^2), reward = potential_curr - potential_next
    #   Positive when distance to pad decreases, zero when stationary.
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 1.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: velocity_damping (proximity_gated quadratic penalty)
    #   Penalise speed more strongly when near the pad.
    # ------------------------------------------------------------
    dist_next_sqrt = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    gate = 1.0 / (1.0 + 2.0 * dist_next_sqrt)          # alpha = 2.0
    vel_penalty = 0.1 * (next_obs[2] ** 2 + next_obs[3] ** 2) * gate
    velocity_damping_reward = -vel_penalty

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    #   Keep body upright and angular velocity small.
    # ------------------------------------------------------------
    angle_penalty = 1.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.2 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: safe_contact_reward (joint_condition_proxy)
    #   Give positive reward when both legs touch the pad,
    #   vehicle is at pad height, and vertical velocity is low.
    # ------------------------------------------------------------
    both_contacts = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    # y position close to zero (pad height)
    proximity_to_pad = max(0.0, 1.0 - abs(next_obs[1]) / 0.2)   # threshold_y = 0.2
    # vertical velocity small
    soft_vel = max(0.0, 1.0 - abs(next_obs[3]) / 0.2)           # threshold_vy = 0.2
    safe_contact_reward = 0.5 * both_contacts * proximity_to_pad * soft_vel

    total_reward = (progress_reward +
                    velocity_damping_reward +
                    orientation_stability_reward +
                    safe_contact_reward)

    components = {
        "progress_reward": progress_reward,
        "velocity_damping_reward": velocity_damping_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "safe_contact_reward": safe_contact_reward,
    }

    return total_reward, components
```

# 历史 iter_02 训练反馈
# Training Feedback

## Final-policy outcome
score=124.271173, len=604.200000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[-87.308558, 244.895182]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_reward | 852.000000 | 98.5% | 98.5% | 7.1% |
| orientation_stability_reward | -9.478931 | -1.1% | 1.1% | 100.0% |
| progress_reward | 1.877989 | 0.2% | 0.2% | 99.2% |
| velocity_damping_reward | -1.435409 | -0.2% | 0.2% | 99.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_02 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 1.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: velocity_damping (proximity_gated quadratic penalty)
    # ------------------------------------------------------------
    dist_next_sqrt = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    gate = 1.0 / (1.0 + 2.0 * dist_next_sqrt)
    vel_penalty = 0.1 * (next_obs[2] ** 2 + next_obs[3] ** 2) * gate
    velocity_damping_reward = -vel_penalty

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    # ------------------------------------------------------------
    angle_penalty = 1.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.2 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: safe_contact_reward (event_based landing bonus)
    #   Reward only the moment both legs first touch the pad.
    #   No continuing reward for staying on the pad.
    # ------------------------------------------------------------
    legs_touching_now = (obs[6] > 0.5) and (obs[7] > 0.5)
    legs_touching_next = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    if (not legs_touching_now) and legs_touching_next:
        safe_contact_reward = 20.0
    else:
        safe_contact_reward = 0.0

    total_reward = (progress_reward +
                    velocity_damping_reward +
                    orientation_stability_reward +
                    safe_contact_reward)

    components = {
        "progress_reward": progress_reward,
        "velocity_damping_reward": velocity_damping_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "safe_contact_reward": safe_contact_reward,
    }

    return total_reward, components
```

# 历史 iter_03 训练反馈
# Training Feedback

## Final-policy outcome
score=110.436884, len=528.550000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-51.736246, 219.059345]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy_reward | 359.974666 | 91.1% | 91.1% | 24.5% |
| action_penalty | -15.682500 | -4.0% | 4.0% | 98.9% |
| orientation_stability_reward | -15.550877 | -3.9% | 3.9% | 100.0% |
| progress_reward | 3.643578 | 0.9% | 1.0% | 99.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_03 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    #   Encourage moving toward the pad center.
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 2.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: landing_proxy (joint_condition_proxy, geometric mean)
    #   Reward simultaneous satisfaction of all landing conditions:
    #   feet contact, centered position, low velocity, low angle,
    #   low angular velocity.
    # ------------------------------------------------------------
    both_contacts = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    # bounded proximity factors: 1/(1 + k*|error|)
    proximity_x = 1.0 / (1.0 + 10.0 * abs(next_obs[0]))   # target radius ~0.1
    proximity_y = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))   # pad height ~0
    low_xvel    = 1.0 / (1.0 + 5.0 * abs(next_obs[2]))    # tolerable x speed
    low_yvel    = 1.0 / (1.0 + 5.0 * abs(next_obs[3]))    # tolerable y speed
    low_angle   = 1.0 / (1.0 + 10.0 * abs(next_obs[4]))   # upright angle
    low_angvel  = 1.0 / (1.0 + 2.0 * abs(next_obs[5]))    # angular velocity

    num_factors = 7
    product = (both_contacts * proximity_x * proximity_y *
               low_xvel * low_yvel * low_angle * low_angvel)
    # geometric mean to prevent collapse of a single factor
    landing_proxy_reward = 4.0 * (product ** (1.0 / num_factors))

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    #   Suppress large angles and angular velocities at all times.
    # ------------------------------------------------------------
    angle_penalty = 2.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.5 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: action_penalty (action_efficiency)
    #   Small fuel penalty for any engine use (actions 1,2,3).
    # ------------------------------------------------------------
    action_penalty = -0.03 * (1.0 if action != 0 else 0.0)

    total_reward = (progress_reward +
                    landing_proxy_reward +
                    orientation_stability_reward +
                    action_penalty)

    components = {
        "progress_reward": progress_reward,
        "landing_proxy_reward": landing_proxy_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "action_penalty": action_penalty,
    }

    return float(total_reward), components
```

# 历史 iter_04 训练反馈
# Training Feedback

## Final-policy outcome
score=227.061788, len=490.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[190.921549, 273.107479]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_reward | 33.088806 | 83.9% | 83.9% | 7.0% |
| orientation_stability_reward | -3.035867 | -7.7% | 7.7% | 100.0% |
| progress_reward | 1.987577 | 5.0% | 5.2% | 97.1% |
| velocity_damping_reward | -1.277038 | -3.2% | 3.2% | 97.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_04 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    #   Potential = -(x^2 + y^2), reward = potential_curr - potential_next
    #   Positive when distance to pad decreases, zero when stationary.
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 1.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: velocity_damping (proximity_gated quadratic penalty)
    #   Penalise speed more strongly when near the pad.
    # ------------------------------------------------------------
    dist_next_sqrt = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    gate = 1.0 / (1.0 + 2.0 * dist_next_sqrt)          # alpha = 2.0
    vel_penalty = 0.1 * (next_obs[2] ** 2 + next_obs[3] ** 2) * gate
    velocity_damping_reward = -vel_penalty

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    #   Keep body upright and angular velocity small.
    # ------------------------------------------------------------
    angle_penalty = 1.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.2 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: safe_contact_reward (joint_condition_proxy)
    #   Give positive reward when both legs touch the pad,
    #   vehicle is at pad height, and vertical velocity is low.
    #   (Coefficient increased from 0.5 to 1.0)
    # ------------------------------------------------------------
    both_contacts = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    # y position close to zero (pad height)
    proximity_to_pad = max(0.0, 1.0 - abs(next_obs[1]) / 0.2)   # threshold_y = 0.2
    # vertical velocity small
    soft_vel = max(0.0, 1.0 - abs(next_obs[3]) / 0.2)           # threshold_vy = 0.2
    safe_contact_reward = 1.0 * both_contacts * proximity_to_pad * soft_vel

    total_reward = (progress_reward +
                    velocity_damping_reward +
                    orientation_stability_reward +
                    safe_contact_reward)

    components = {
        "progress_reward": progress_reward,
        "velocity_damping_reward": velocity_damping_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "safe_contact_reward": safe_contact_reward,
    }

    return float(total_reward), components
```

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | orientation_stability_reward + progress_reward + safe_contact_reward + velocity_damping_reward | 144.17 | 144.17 | 0.00 | 965.45 | orientation_stability_reward=-0.019 progress_reward=0.005 safe_contact_reward=0.278 velocity_damping_reward=-0.007 | new_best |
| 2 | orientation_stability_reward + progress_reward + safe_contact_reward + velocity_damping_reward | 124.27 | 144.17 | -19.90 | 604.20 | orientation_stability_reward=-0.030 progress_reward=0.008 safe_contact_reward=1.037 velocity_damping_reward=-0.013 | no_meaningful_improvement |
| 3 | action_penalty + landing_proxy_reward + orientation_stability_reward + progress_reward | 110.44 | 144.17 | -33.74 | 528.55 | action_penalty=-0.020 landing_proxy_reward=1.192 orientation_stability_reward=-0.078 progress_reward=0.021 | no_meaningful_improvement |
| 4 | orientation_stability_reward + progress_reward + safe_contact_reward + velocity_damping_reward | 227.06 | 227.06 | 0.00 | 490.10 | orientation_stability_reward=-0.020 progress_reward=0.005 safe_contact_reward=0.476 velocity_damping_reward=-0.008 | target_solved_new_best |

```
