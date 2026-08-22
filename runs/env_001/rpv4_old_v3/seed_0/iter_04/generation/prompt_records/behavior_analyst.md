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
# 本轮训练反馈 (score=212.944108)
# Training Feedback

## Final-policy outcome
score=212.944108, len=511.900000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[69.659258, 274.047242]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_reward | 37.250000 | 90.7% | 90.7% | 7.3% |
| goal_proximity | 2.638238 | 6.4% | 6.8% | 97.6% |
| living_penalty | -1.023800 | -2.5% | 2.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估数据（每10%步数的官方环境得分）
| pct | score_mean | score_std |
|---|---|---|
| 40% | 0.00 | 0.00 |
| 80% | 0.00 | 0.00 |

# 本轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离计算
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（距离改善差分）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)

    # 2. 着陆成功奖励
    landing_dist_thresh = 0.2
    landing_vel_thresh = 0.15
    landing_ang_thresh = 0.1
    w_landing = 1.0

    contacts = left_contact + right_contact
    near_center = next_dist < landing_dist_thresh
    low_speed = abs(next_vx) < landing_vel_thresh and abs(next_vy) < landing_vel_thresh
    upright = abs(body_angle) < landing_ang_thresh

    landing_success = (contacts == 2) and near_center and low_speed and upright
    reward_landing = w_landing if landing_success else 0.0

    # 3. 时间惩罚
    living_penalty = -0.002

    total_reward = reward_prox + reward_landing + living_penalty

    components = {
        "goal_proximity": reward_prox,
        "landing_success_reward": reward_landing,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```

# 历史 iter_01 训练反馈
# Training Feedback

## Final-policy outcome
score=-40.037266, len=154.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-133.278295, 15.315072]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 2.138831 | 80.7% | 86.4% | 100.0% |
| living_penalty | -0.308100 | -11.6% | 11.6% | 100.0% |
| soft_landing_penalty | -0.052633 | -2.0% | 2.0% | 1.5% |
| crash_prevention_x | 0.000000 | 0.0% | 0.0% | 0.0% |
| crash_prevention_y | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_01 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v1: 主进展信号 + 软着陆约束 + 碰撞预防 + 时间惩罚
    """
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]

    # 距离计算（欧几里得距离）
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（基于距离改善的差分奖励）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)  # 距离减小得正奖励

    # 2. 软着陆约束：接近垫子时惩罚高速和倾斜姿态
    w_soft = 0.5
    proximity_threshold = 0.3  # 当距离 < 0.3 时激活
    gate = max(0.0, 1.0 - next_dist / proximity_threshold)  # 线性衰减门控
    # 速度用绝对值，姿态用平方（抑制大幅倾斜）
    soft_penalty_intensity = abs(next_vx) + abs(next_vy) + body_angle ** 2
    penalty_soft = -w_soft * gate * soft_penalty_intensity

    # 3. 碰撞预防：水平越界惩罚
    x_bound = 0.9  # 视野假设边界
    w_crash_x = 10.0
    penalty_crash_x = 0.0
    if abs(next_x) > x_bound:
        penalty_crash_x = -w_crash_x * (abs(next_x) - x_bound) ** 2

    # 4. 碰撞预防：高速垂直坠落惩罚（低于垫子且下落过快）
    danger_vy = -0.5
    w_crash_y = 5.0
    penalty_crash_y = 0.0
    if next_y < -0.1 and next_vy < danger_vy:  # 低于垫子且速度向下超过阈值
        # next_vy 为负值，差值平方惩罚强度随速度增大而增加
        penalty_crash_y = -w_crash_y * (next_vy - danger_vy) ** 2

    # 5. 时间惩罚：推动快速完成任务
    living_penalty = -0.002

    total_reward = (reward_prox + penalty_soft +
                    penalty_crash_x + penalty_crash_y +
                    living_penalty)

    components = {
        "goal_proximity": reward_prox,
        "soft_landing_penalty": penalty_soft,
        "crash_prevention_x": penalty_crash_x,
        "crash_prevention_y": penalty_crash_y,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```

# 历史 iter_02 训练反馈
# Training Feedback

## Final-policy outcome
score=217.212007, len=484.100000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[61.623931, 265.546415]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_reward | 36.450000 | 90.6% | 90.6% | 7.5% |
| goal_proximity | 2.567816 | 6.4% | 7.0% | 97.6% |
| living_penalty | -0.968200 | -2.4% | 2.4% | 100.0% |
| crash_prevention_x | 0.000000 | 0.0% | 0.0% | 0.0% |
| crash_prevention_y | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_02 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    # 腿接触信息（0/1）
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离计算
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（距离改善差分）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)

    # 2. 着陆成功奖励（替代原 soft_landing_penalty）
    # 当双腿接触、位置接近原点、速度小、姿态水平时给予持续正奖励
    landing_dist_thresh = 0.2
    landing_vel_thresh = 0.15   # 用于水平/垂直线速度绝对值
    landing_ang_thresh = 0.1    # 角度绝对值
    w_landing = 1.0

    contacts = left_contact + right_contact           # 0, 1, 2
    near_center = next_dist < landing_dist_thresh
    low_speed = abs(next_vx) < landing_vel_thresh and abs(next_vy) < landing_vel_thresh
    upright = abs(body_angle) < landing_ang_thresh

    # 只有两条腿都与地面接触且满足其他条件才算成功着陆
    landing_success = (contacts == 2) and near_center and low_speed and upright
    reward_landing = w_landing if landing_success else 0.0

    # 3. 碰撞预防（保留原有逻辑，尝试微调使之可能触发）
    x_bound = 0.9
    w_crash_x = 10.0
    penalty_crash_x = 0.0
    if abs(next_x) > x_bound:
        penalty_crash_x = -w_crash_x * (abs(next_x) - x_bound) ** 2

    danger_vy = -0.5
    w_crash_y = 5.0
    penalty_crash_y = 0.0
    if next_y < -0.1 and next_vy < danger_vy:
        penalty_crash_y = -w_crash_y * (next_vy - danger_vy) ** 2

    # 4. 时间惩罚
    living_penalty = -0.002

    total_reward = (reward_prox + reward_landing +
                    penalty_crash_x + penalty_crash_y +
                    living_penalty)

    components = {
        "goal_proximity": reward_prox,
        "landing_success_reward": reward_landing,
        "crash_prevention_x": penalty_crash_x,
        "crash_prevention_y": penalty_crash_y,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```

# 历史 iter_03 训练反馈
# Training Feedback

## Final-policy outcome
score=212.944108, len=511.900000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[69.659258, 274.047242]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_reward | 37.250000 | 90.7% | 90.7% | 7.3% |
| goal_proximity | 2.638238 | 6.4% | 6.8% | 97.6% |
| living_penalty | -1.023800 | -2.5% | 2.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_03 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离计算
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（距离改善差分）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)

    # 2. 着陆成功奖励
    landing_dist_thresh = 0.2
    landing_vel_thresh = 0.15
    landing_ang_thresh = 0.1
    w_landing = 1.0

    contacts = left_contact + right_contact
    near_center = next_dist < landing_dist_thresh
    low_speed = abs(next_vx) < landing_vel_thresh and abs(next_vy) < landing_vel_thresh
    upright = abs(body_angle) < landing_ang_thresh

    landing_success = (contacts == 2) and near_center and low_speed and upright
    reward_landing = w_landing if landing_success else 0.0

    # 3. 时间惩罚
    living_penalty = -0.002

    total_reward = reward_prox + reward_landing + living_penalty

    components = {
        "goal_proximity": reward_prox,
        "landing_success_reward": reward_landing,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crash_prevention_x + crash_prevention_y + goal_proximity + living_penalty + soft_landing_penalty | -40.04 | -40.04 | 0.00 | 154.05 | crash_prevention_x=-0.000 crash_prevention_y=-0.007 goal_proximity=0.021 living_penalty=-0.002 soft_landing_penalty=-0.012 | new_best |
| 2 | crash_prevention_x + crash_prevention_y + goal_proximity + landing_success_reward + living_penalty | 217.21 | 217.21 | 0.00 | 484.10 | crash_prevention_x=-0.000 crash_prevention_y=-0.001 goal_proximity=0.013 landing_success_reward=0.312 living_penalty=-0.002 | target_solved_new_best |
| 3 | goal_proximity + landing_success_reward + living_penalty | 212.94 | 217.21 | -4.27 | 511.90 | goal_proximity=0.012 landing_success_reward=0.348 living_penalty=-0.002 | target_solved_no_improvement |


# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    # 腿接触信息（0/1）
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离计算
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（距离改善差分）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)

    # 2. 着陆成功奖励（替代原 soft_landing_penalty）
    # 当双腿接触、位置接近原点、速度小、姿态水平时给予持续正奖励
    landing_dist_thresh = 0.2
    landing_vel_thresh = 0.15   # 用于水平/垂直线速度绝对值
    landing_ang_thresh = 0.1    # 角度绝对值
    w_landing = 1.0

    contacts = left_contact + right_contact           # 0, 1, 2
    near_center = next_dist < landing_dist_thresh
    low_speed = abs(next_vx) < landing_vel_thresh and abs(next_vy) < landing_vel_thresh
    upright = abs(body_angle) < landing_ang_thresh

    # 只有两条腿都与地面接触且满足其他条件才算成功着陆
    landing_success = (contacts == 2) and near_center and low_speed and upright
    reward_landing = w_landing if landing_success else 0.0

    # 3. 碰撞预防（保留原有逻辑，尝试微调使之可能触发）
    x_bound = 0.9
    w_crash_x = 10.0
    penalty_crash_x = 0.0
    if abs(next_x) > x_bound:
        penalty_crash_x = -w_crash_x * (abs(next_x) - x_bound) ** 2

    danger_vy = -0.5
    w_crash_y = 5.0
    penalty_crash_y = 0.0
    if next_y < -0.1 and next_vy < danger_vy:
        penalty_crash_y = -w_crash_y * (next_vy - danger_vy) ** 2

    # 4. 时间惩罚
    living_penalty = -0.002

    total_reward = (reward_prox + reward_landing +
                    penalty_crash_x + penalty_crash_y +
                    living_penalty)

    components = {
        "goal_proximity": reward_prox,
        "landing_success_reward": reward_landing,
        "crash_prevention_x": penalty_crash_x,
        "crash_prevention_y": penalty_crash_y,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```
```
