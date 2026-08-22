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
# 本轮训练反馈 (score=203.999462)
# Training Feedback

## Final-policy outcome
score=203.999462, len=569.650000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[68.170546, 245.278434]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_completion | 363.115059 | 79.1% | 79.1% | 16.5% |
| soft_landing_speed_penalty | -71.365811 | -15.6% | 15.6% | 99.8% |
| approach_progress | 19.318184 | 4.2% | 4.5% | 97.8% |
| orientation_penalty | -3.520726 | -0.8% | 0.8% | 100.0% |

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
    # Unpack current and next observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 1. Target approach: improvement delta ----------
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = dist_curr - dist_next           # positive when agent moves closer
    w_approach = 15.0                          # increased from 5.0 to amplify approach guidance
    reward_approach = w_approach * progress

    # ---------- 2. Soft landing speed: distance-gated velocity penalty ----------
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    sigma = 0.5                                # determines how sharply gate rises near target
    gate = 2.718281828 ** (-dist_next**2 / sigma**2)
    w_speed = 2.0
    reward_speed = -w_speed * speed_next * gate

    # ---------- 3. Stable orientation: quadratic penalties on tilt and rotation ----------
    w_angle = 1.0
    w_angvel = 0.1
    reward_orientation = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # ---------- 4. Contact completion: dual-leg contact gated by position and speed ----------
    left = left_contact_next
    right = right_contact_next
    beta_dist = 10.0
    beta_speed = 10.0
    # Soft proximity factors in [0,1] – near 1 only when distance/speed are near zero
    dist_factor = 1.0 / (1.0 + beta_dist * dist_next**2)
    speed_factor = 1.0 / (1.0 + beta_speed * speed_next**2)
    # Additive contact to give partial reward for single-leg touchdowns
    w_contact = 2.5
    reward_contact = w_contact * (left + right) * dist_factor * speed_factor

    total = reward_approach + reward_speed + reward_orientation + reward_contact

    components = {
        'approach_progress': reward_approach,
        'soft_landing_speed_penalty': reward_speed,
        'orientation_penalty': reward_orientation,
        'contact_completion': reward_contact,
    }
    return float(total), components
```

# 历史 iter_01 训练反馈
# Training Feedback

## Final-policy outcome
score=81.184393, len=890.750000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-27.608441, 204.846063]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_completion | 478.927441 | 84.8% | 84.8% | 15.8% |
| soft_landing_speed_penalty | -73.990109 | -13.1% | 13.1% | 100.0% |
| approach_progress | 5.892979 | 1.0% | 1.2% | 100.0% |
| orientation_penalty | -5.213019 | -0.9% | 0.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_01 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 1. Target approach: improvement delta ----------
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = dist_curr - dist_next           # positive when agent moves closer
    w_approach = 5.0
    reward_approach = w_approach * progress

    # ---------- 2. Soft landing speed: distance‑gated velocity penalty ----------
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    sigma = 0.5                                # determines how sharply gate rises near target
    # Exponential gate: close to 1 only when dist_next is small
    gate = 2.718281828 ** (-dist_next**2 / sigma**2)
    w_speed = 2.0
    reward_speed = -w_speed * speed_next * gate

    # ---------- 3. Stable orientation: quadratic penalties on tilt and rotation ----------
    w_angle = 1.0
    w_angvel = 0.1
    reward_orientation = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # ---------- 4. Contact completion: dual‑leg contact gated by position and speed ----------
    left = left_contact_next
    right = right_contact_next
    beta_dist = 10.0
    beta_speed = 10.0
    # Soft proximity factors in [0,1] – near 1 only when distance/speed are near zero
    dist_factor = 1.0 / (1.0 + beta_dist * dist_next**2)
    speed_factor = 1.0 / (1.0 + beta_speed * speed_next**2)
    w_contact = 5.0
    reward_contact = w_contact * left * right * dist_factor * speed_factor

    total = reward_approach + reward_speed + reward_orientation + reward_contact

    components = {
        'approach_progress': reward_approach,
        'soft_landing_speed_penalty': reward_speed,
        'orientation_penalty': reward_orientation,
        'contact_completion': reward_contact,
    }
    return float(total), components
```

# 历史 iter_02 训练反馈
# Training Feedback

## Final-policy outcome
score=215.504114, len=517.900000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[79.062130, 258.138414]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_completion | 356.972250 | 83.6% | 83.6% | 18.4% |
| soft_landing_speed_penalty | -60.296305 | -14.1% | 14.1% | 99.8% |
| approach_progress | 6.596181 | 1.5% | 1.6% | 97.7% |
| orientation_penalty | -2.671592 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_02 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 1. Target approach: improvement delta ----------
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = dist_curr - dist_next           # positive when agent moves closer
    w_approach = 5.0
    reward_approach = w_approach * progress

    # ---------- 2. Soft landing speed: distance-gated velocity penalty ----------
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    sigma = 0.5                                # determines how sharply gate rises near target
    gate = 2.718281828 ** (-dist_next**2 / sigma**2)
    w_speed = 2.0
    reward_speed = -w_speed * speed_next * gate

    # ---------- 3. Stable orientation: quadratic penalties on tilt and rotation ----------
    w_angle = 1.0
    w_angvel = 0.1
    reward_orientation = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # ---------- 4. Contact completion: dual-leg contact gated by position and speed ----------
    left = left_contact_next
    right = right_contact_next
    beta_dist = 10.0
    beta_speed = 10.0
    # Soft proximity factors in [0,1] – near 1 only when distance/speed are near zero
    dist_factor = 1.0 / (1.0 + beta_dist * dist_next**2)
    speed_factor = 1.0 / (1.0 + beta_speed * speed_next**2)
    # Modified: additive contact to give partial reward for single-leg touchdowns
    w_contact = 2.5
    reward_contact = w_contact * (left + right) * dist_factor * speed_factor

    total = reward_approach + reward_speed + reward_orientation + reward_contact

    components = {
        'approach_progress': reward_approach,
        'soft_landing_speed_penalty': reward_speed,
        'orientation_penalty': reward_orientation,
        'contact_completion': reward_contact,
    }
    return float(total), components
```

# 历史 iter_03 训练反馈
# Training Feedback

## Final-policy outcome
score=203.999462, len=569.650000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[68.170546, 245.278434]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_completion | 363.115059 | 79.1% | 79.1% | 16.5% |
| soft_landing_speed_penalty | -71.365811 | -15.6% | 15.6% | 99.8% |
| approach_progress | 19.318184 | 4.2% | 4.5% | 97.8% |
| orientation_penalty | -3.520726 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_03 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 1. Target approach: improvement delta ----------
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = dist_curr - dist_next           # positive when agent moves closer
    w_approach = 15.0                          # increased from 5.0 to amplify approach guidance
    reward_approach = w_approach * progress

    # ---------- 2. Soft landing speed: distance-gated velocity penalty ----------
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    sigma = 0.5                                # determines how sharply gate rises near target
    gate = 2.718281828 ** (-dist_next**2 / sigma**2)
    w_speed = 2.0
    reward_speed = -w_speed * speed_next * gate

    # ---------- 3. Stable orientation: quadratic penalties on tilt and rotation ----------
    w_angle = 1.0
    w_angvel = 0.1
    reward_orientation = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # ---------- 4. Contact completion: dual-leg contact gated by position and speed ----------
    left = left_contact_next
    right = right_contact_next
    beta_dist = 10.0
    beta_speed = 10.0
    # Soft proximity factors in [0,1] – near 1 only when distance/speed are near zero
    dist_factor = 1.0 / (1.0 + beta_dist * dist_next**2)
    speed_factor = 1.0 / (1.0 + beta_speed * speed_next**2)
    # Additive contact to give partial reward for single-leg touchdowns
    w_contact = 2.5
    reward_contact = w_contact * (left + right) * dist_factor * speed_factor

    total = reward_approach + reward_speed + reward_orientation + reward_contact

    components = {
        'approach_progress': reward_approach,
        'soft_landing_speed_penalty': reward_speed,
        'orientation_penalty': reward_orientation,
        'contact_completion': reward_contact,
    }
    return float(total), components
```

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | approach_progress + contact_completion + orientation_penalty + soft_landing_speed_penalty | 81.18 | 81.18 | 0.00 | 890.75 | approach_progress=0.010 contact_completion=1.878 orientation_penalty=-0.020 soft_landing_speed_penalty=-0.080 | new_best |
| 2 | approach_progress + contact_completion + orientation_penalty + soft_landing_speed_penalty | 215.50 | 215.50 | 0.00 | 517.90 | approach_progress=0.010 contact_completion=1.537 orientation_penalty=-0.029 soft_landing_speed_penalty=-0.082 | target_solved_new_best |
| 3 | approach_progress + contact_completion + orientation_penalty + soft_landing_speed_penalty | 204.00 | 215.50 | -11.50 | 569.65 | approach_progress=0.037 contact_completion=1.809 orientation_penalty=-0.026 soft_landing_speed_penalty=-0.092 | target_solved_no_improvement |


# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 1. Target approach: improvement delta ----------
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = dist_curr - dist_next           # positive when agent moves closer
    w_approach = 5.0
    reward_approach = w_approach * progress

    # ---------- 2. Soft landing speed: distance-gated velocity penalty ----------
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    sigma = 0.5                                # determines how sharply gate rises near target
    gate = 2.718281828 ** (-dist_next**2 / sigma**2)
    w_speed = 2.0
    reward_speed = -w_speed * speed_next * gate

    # ---------- 3. Stable orientation: quadratic penalties on tilt and rotation ----------
    w_angle = 1.0
    w_angvel = 0.1
    reward_orientation = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # ---------- 4. Contact completion: dual-leg contact gated by position and speed ----------
    left = left_contact_next
    right = right_contact_next
    beta_dist = 10.0
    beta_speed = 10.0
    # Soft proximity factors in [0,1] – near 1 only when distance/speed are near zero
    dist_factor = 1.0 / (1.0 + beta_dist * dist_next**2)
    speed_factor = 1.0 / (1.0 + beta_speed * speed_next**2)
    # Modified: additive contact to give partial reward for single-leg touchdowns
    w_contact = 2.5
    reward_contact = w_contact * (left + right) * dist_factor * speed_factor

    total = reward_approach + reward_speed + reward_orientation + reward_contact

    components = {
        'approach_progress': reward_approach,
        'soft_landing_speed_penalty': reward_speed,
        'orientation_penalty': reward_orientation,
        'contact_completion': reward_contact,
    }
    return float(total), components
```
```
