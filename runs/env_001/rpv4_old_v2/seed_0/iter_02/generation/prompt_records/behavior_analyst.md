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
# 本轮训练反馈 (score=225.882125)
# Training Feedback

## Final-policy outcome
score=225.882125, len=479.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[199.061371, 261.871767]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 346.757918 | 73.2% | 73.2% | 100.0% |
| contact_reward | 70.211244 | 14.8% | 14.8% | 7.8% |
| landing_speed_penalty | -56.494578 | -11.9% | 11.9% | 99.8% |
| settlement_penalty | -0.333686 | -0.1% | 0.1% | 7.8% |

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
    """
    v1 reward function for the soft landing / docking task.

    Roles:
      - approach_target: continuous distance reward (bounded, positive)
      - safe_landing: speed penalty near pad + contact bonus for both legs
      - settlement_stabilization: penalty for velocity/angular movement after contact
    """
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # helpers
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    both_contacts = (left_contact > 0.5) and (right_contact > 0.5)

    # component A: approach_target – positive bounded distance reward
    # using 1/(1 + kp*dist) → max 1.0 when exactly on target
    kp = 1.0
    w_progress = 1.0
    progress_reward = w_progress * (1.0 / (1.0 + kp * dist))

    # component B: safe_landing – speed penalty when close to pad, contact bonus
    threshold_speed = 2.0        # apply speed penalty only within this distance
    w_speed = 0.5                # penalty weight for |vx|+|vy|
    if dist < threshold_speed:
        landing_speed_penalty = -w_speed * (abs(x_vel) + abs(y_vel))
    else:
        landing_speed_penalty = 0.0

    # contact bonus: both legs touching, higher when near center
    w_contact = 2.0
    kc = 1.0
    if both_contacts:
        contact_reward = w_contact * (1.0 / (1.0 + kc * dist))
    else:
        contact_reward = 0.0

    # component C: settlement_stabilization – after contact, encourage full stillness
    w_settle = 0.2
    if both_contacts:
        settlement_penalty = -w_settle * (
            abs(x_vel) + abs(y_vel) + abs(angular_vel) + abs(body_angle)
        )
    else:
        settlement_penalty = 0.0

    total_reward = progress_reward + landing_speed_penalty + contact_reward + settlement_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_speed_penalty": landing_speed_penalty,
        "contact_reward": contact_reward,
        "settlement_penalty": settlement_penalty,
    }

    return float(total_reward), components
```

# 历史 iter_01 训练反馈
# Training Feedback

## Final-policy outcome
score=225.882125, len=479.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[199.061371, 261.871767]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 346.757918 | 73.2% | 73.2% | 100.0% |
| contact_reward | 70.211244 | 14.8% | 14.8% | 7.8% |
| landing_speed_penalty | -56.494578 | -11.9% | 11.9% | 99.8% |
| settlement_penalty | -0.333686 | -0.1% | 0.1% | 7.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_01 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward function for the soft landing / docking task.

    Roles:
      - approach_target: continuous distance reward (bounded, positive)
      - safe_landing: speed penalty near pad + contact bonus for both legs
      - settlement_stabilization: penalty for velocity/angular movement after contact
    """
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # helpers
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    both_contacts = (left_contact > 0.5) and (right_contact > 0.5)

    # component A: approach_target – positive bounded distance reward
    # using 1/(1 + kp*dist) → max 1.0 when exactly on target
    kp = 1.0
    w_progress = 1.0
    progress_reward = w_progress * (1.0 / (1.0 + kp * dist))

    # component B: safe_landing – speed penalty when close to pad, contact bonus
    threshold_speed = 2.0        # apply speed penalty only within this distance
    w_speed = 0.5                # penalty weight for |vx|+|vy|
    if dist < threshold_speed:
        landing_speed_penalty = -w_speed * (abs(x_vel) + abs(y_vel))
    else:
        landing_speed_penalty = 0.0

    # contact bonus: both legs touching, higher when near center
    w_contact = 2.0
    kc = 1.0
    if both_contacts:
        contact_reward = w_contact * (1.0 / (1.0 + kc * dist))
    else:
        contact_reward = 0.0

    # component C: settlement_stabilization – after contact, encourage full stillness
    w_settle = 0.2
    if both_contacts:
        settlement_penalty = -w_settle * (
            abs(x_vel) + abs(y_vel) + abs(angular_vel) + abs(body_angle)
        )
    else:
        settlement_penalty = 0.0

    total_reward = progress_reward + landing_speed_penalty + contact_reward + settlement_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_speed_penalty": landing_speed_penalty,
        "contact_reward": contact_reward,
        "settlement_penalty": settlement_penalty,
    }

    return float(total_reward), components
```

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_reward + landing_speed_penalty + progress_reward + settlement_penalty | 225.88 | 225.88 | 0.00 | 479.70 | contact_reward=0.789 landing_speed_penalty=-0.114 progress_reward=0.717 settlement_penalty=-0.010 | target_solved_new_best |

```
