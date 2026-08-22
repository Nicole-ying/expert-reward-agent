# Prompt Record

## System Prompt

```text
你是训练行为诊断分析师。你的唯一职责是：阅读训练反馈和迭代历史，诊断策略学到了什么行为、哪个组件需要干预。

你**不设计 reward、不写代码、不选干预层级**。你只产出诊断报告。

# 输入数据

1. **本轮训练反馈**：score, score_range, len, terminated/truncated, 组件表
2. **本轮 reward 代码**
3. **上一轮训练反馈 + reward 代码**（如存在，用于对比修改效果）
4. **历史记忆表**
5. **环境事实摘要**：对照任务目标，看每个组件的量级是否合理。如果完成信号比过程引导信号大几个数量级，奖励结构和任务目标是错位的。

# 关键原则

- **不同组件有不同的天然 profile。** 完成信号 active_rate 低是正常的（着陆只在末尾），稠密 progress 信号 active_rate 应接近 100%。不要用一个阈值套所有组件。
- **趋势比快照重要。** score 大涨说明修改方向对，即使数字看起来奇怪。terminated 率高可能是 crash 也可能是成功着陆，结合 score 和 len 判断。
- **per-step 量级比 share% 更直观。** episode_sum_mean ÷ len 就是每步均值。对比两个组件的 per-step 量级，比看 magnitude_share 更容易发现错位。
- **看谁被淹没，不止看谁最大。** 一个 100% active 但量级极小的 progress 信号，比一个占 90% share 的 contact 信号更需要干预——前者是"指南针失效"。

# 输出格式

## 1. 当前策略行为画像
[agent 学会了什么？引用具体数字。1-2 句话。]

## 2. 组件健康度
[每个组件一行判定]
- component_a: [健康 / 死组件 / 主导但错位 / 淹没 / 全覆盖惩罚]
- component_b: ...

## 3. 骨架演进表（必输出）

如果存在多轮数据，并排展示同一骨架下各轮指标。这是你最重要的产出。
```
| iter | score | len | term | A_ep_sum | A_active | B_ep_sum | B_active | ... |
| N-2  |       |     |      |          |          |          |          |     |
| N-1  |       |     |      |          |          |          |          |     |
| N    |       |     |      |          |          |          |          |     |
```
表后 1-2 句话：哪个组件被改动了？产生什么连锁反应？交互效应是什么？

## 4. 行为结论
[健康且稳定 / 达标但不稳定 / 病态]
[最需要干预的组件及原因]

```

## User Prompt

```markdown
# 本轮训练反馈 (score=17.867905)
# Training Feedback

## Final-policy outcome
score=17.867905, len=784.350000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-169.804912, 227.300020]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_reward | 509.239245 | 90.1% | 90.1% | 15.9% |
| velocity_brake_near_target | -32.704433 | -5.8% | 5.8% | 100.0% |
| goal_approach_progress | 8.423406 | 1.5% | 2.5% | 100.0% |
| orientation_stability | -9.012481 | -1.6% | 1.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 本轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation and soft landing task.
    
    Task: Navigate from random perturbed initial position to target landing pad,
    achieve stable low-speed contact with upright orientation.
    
    Components:
    - goal_approach_progress: dense shaping via distance improvement delta
    - velocity_brake_near_target: quadratic velocity penalty scaled by proximity
    - orientation_stability: quadratic penalty on angle and angular velocity
    - contact_landing_reward: continuous contact reward with ADDITIVE quality averaging
    """
    
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle, next_ang_vel = next_obs[4], next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    components = {}
    
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (next_x_pos**2 + next_y_pos**2) ** 0.5
    distance_delta = current_distance - next_distance
    components["goal_approach_progress"] = 10.0 * distance_delta
    
    velocity_magnitude = (next_x_vel**2 + next_y_vel**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + next_distance)
    components["velocity_brake_near_target"] = -2.0 * proximity_factor * velocity_magnitude**2
    
    angle_penalty = next_angle**2
    ang_vel_penalty = next_ang_vel**2
    components["orientation_stability"] = -1.0 * (angle_penalty + 0.5 * ang_vel_penalty)
    
    contact_active = max(next_left_contact, next_right_contact)
    
    velocity_quality = max(0.0, 1.0 - velocity_magnitude / 1.0)
    angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)
    proximity_quality = max(0.0, 1.0 - next_distance / 1.0)
    
    quality_factor = (velocity_quality + angle_quality + proximity_quality) / 3.0
    components["contact_landing_reward"] = 5.0 * contact_active * quality_factor
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```

# 上一轮训练反馈 (score=17.867905)
# Training Feedback

## Final-policy outcome
score=17.867905, len=784.350000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-169.804912, 227.300020]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_reward | 509.239245 | 90.1% | 90.1% | 15.9% |
| velocity_brake_near_target | -32.704433 | -5.8% | 5.8% | 100.0% |
| goal_approach_progress | 8.423406 | 1.5% | 2.5% | 100.0% |
| orientation_stability | -9.012481 | -1.6% | 1.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation and soft landing task.
    
    Task: Navigate from random perturbed initial position to target landing pad,
    achieve stable low-speed contact with upright orientation.
    
    Components:
    - goal_approach_progress: dense shaping via distance improvement delta
    - velocity_brake_near_target: quadratic velocity penalty scaled by proximity
    - orientation_stability: quadratic penalty on angle and angular velocity
    - contact_landing_reward: continuous contact reward with ADDITIVE quality averaging
    """
    
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle, next_ang_vel = next_obs[4], next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    components = {}
    
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (next_x_pos**2 + next_y_pos**2) ** 0.5
    distance_delta = current_distance - next_distance
    components["goal_approach_progress"] = 10.0 * distance_delta
    
    velocity_magnitude = (next_x_vel**2 + next_y_vel**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + next_distance)
    components["velocity_brake_near_target"] = -2.0 * proximity_factor * velocity_magnitude**2
    
    angle_penalty = next_angle**2
    ang_vel_penalty = next_ang_vel**2
    components["orientation_stability"] = -1.0 * (angle_penalty + 0.5 * ang_vel_penalty)
    
    contact_active = max(next_left_contact, next_right_contact)
    
    velocity_quality = max(0.0, 1.0 - velocity_magnitude / 1.0)
    angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)
    proximity_quality = max(0.0, 1.0 - next_distance / 1.0)
    
    quality_factor = (velocity_quality + angle_quality + proximity_quality) / 3.0
    components["contact_landing_reward"] = 5.0 * contact_active * quality_factor
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```

# 逐组件修改效果（编辑前 → 编辑后）
# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| contact_landing_reward | 2.291 | 2.102 | -0.189 | 54.0% | 50.3% | -3.8pp |
| goal_approach_progress | 0.230 | 0.030 | -0.200 | 100.0% | 100.0% | -0.0pp |
| orientation_stability | -0.054 | -0.064 | -0.010 | 100.0% | 100.0% | +0.0pp |
| velocity_brake_near_target | -0.183 | -0.202 | -0.019 | 100.0% | 100.0% | -0.0pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -38.36 | 17.87 | +56.23 |
| 平均回合长度 | 951.8 | 784.4 | -167.4 |
| 终止回合数 | 3/20 | 11/20 | +8 |

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 984.85 | -82.86 | ✅ |
| 2 | 骨架变化: contact_landing_reward + goal_approach_progress +  | — | 721.10 | 142.32 | ✅ |
| 3 | 骨架变化: contact_landing_reward + goal_approach_progress +  | — | 930.15 | 36.97 | ❌ |
| 4 | 骨架变化: contact_landing_reward + goal_approach_progress +  | — | 736.95 | 103.63 | ❌ |
| 5 | 骨架变化: contact_landing_reward + goal_approach_progress +  | — | 951.80 | -38.36 | ➖ |
| 6 | 骨架变化: contact_landing_reward + goal_approach_progress +  | — | 784.35 | 17.87 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | goal_approach_progress + orientation_stability + terminal_soft_landing_bonus + velocity_brake_near_target | -82.86 | -82.86 | 0.00 | 984.85 | goal_approach_progress=0.031 orientation_stability=-0.071 terminal_soft_landing_bonus=14.147 velocity_brake_near_target=-0.200 | new_best |
| 2 | contact_landing_reward + goal_approach_progress + orientation_stability + velocity_brake_near_target | 142.32 | 142.32 | 0.00 | 721.10 | contact_landing_reward=2.104 goal_approach_progress=0.027 orientation_stability=-0.062 velocity_brake_near_target=-0.175 | new_best |
| 3 | contact_landing_reward + goal_approach_progress + orientation_stability + velocity_brake_near_target | 36.97 | 142.32 | -105.35 | 930.15 | contact_landing_reward=2.183 goal_approach_progress=0.171 orientation_stability=-0.066 velocity_brake_near_target=-0.238 | no_meaningful_improvement |
| 4 | contact_landing_reward + goal_approach_progress + orientation_stability + velocity_brake_near_target | 103.63 | 142.32 | -38.69 | 736.95 | contact_landing_reward=2.131 goal_approach_progress=0.032 orientation_stability=-0.072 velocity_brake_near_target=-0.223 | no_meaningful_improvement |
| 5 | contact_landing_reward + goal_approach_progress + orientation_stability + velocity_brake_near_target | -38.36 | 142.32 | -180.68 | 951.80 | contact_landing_reward=2.291 goal_approach_progress=0.230 orientation_stability=-0.054 velocity_brake_near_target=-0.183 | unsolved_high_achievement_continue_from_best |
| 6 | contact_landing_reward + goal_approach_progress + orientation_stability + velocity_brake_near_target | 17.87 | 142.32 | -124.45 | 784.35 | contact_landing_reward=2.102 goal_approach_progress=0.030 orientation_stability=-0.064 velocity_brake_near_target=-0.202 | no_meaningful_improvement |


# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation and soft landing task.
    
    Task: Navigate from random perturbed initial position to target landing pad,
    achieve stable low-speed contact with upright orientation.
    
    Components:
    - goal_approach_progress: dense shaping via distance improvement delta
    - velocity_brake_near_target: quadratic velocity penalty scaled by proximity
    - orientation_stability: quadratic penalty on angle and angular velocity
    - contact_landing_reward: continuous contact reward modulated by quality factors
    """
    
    # Extract observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle, next_ang_vel = next_obs[4], next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    components = {}
    
    # === Component A: goal_approach_progress (main learning signal) ===
    # Use potential-based shaping: reward improvement in distance to target
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (next_x_pos**2 + next_y_pos**2) ** 0.5
    distance_delta = current_distance - next_distance  # positive when approaching
    components["goal_approach_progress"] = 10.0 * distance_delta
    
    # === Component B: velocity_brake_near_target (safety constraint) ===
    # Quadratic velocity penalty, scaled by proximity to encourage slow approach
    velocity_magnitude = (next_x_vel**2 + next_y_vel**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + next_distance)  # approaches 1 when close to target
    components["velocity_brake_near_target"] = -2.0 * proximity_factor * velocity_magnitude**2
    
    # === Component C: orientation_stability (safety constraint) ===
    # Quadratic penalty on angle deviation and angular velocity
    angle_penalty = next_angle**2
    ang_vel_penalty = next_ang_vel**2
    components["orientation_stability"] = -1.0 * (angle_penalty + 0.5 * ang_vel_penalty)
    
    # === Component D: contact_landing_reward (dense contact guidance) ===
    # Continuous reward for contact, modulated by landing quality factors
    # Base: any contact (left or right)
    contact_active = max(next_left_contact, next_right_contact)
    
    # Quality modulator 1: velocity (lower is better, bounded to [0,1])
    velocity_quality = max(0.0, 1.0 - velocity_magnitude / 1.0)
    
    # Quality modulator 2: orientation (smaller angle is better, bounded to [0,1])
    angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)
    
    # Quality modulator 3: proximity to target (closer is better, bounded to [0,1])
    proximity_quality = max(0.0, 1.0 - next_distance / 1.0)
    
    # Combined: contact reward scaled by quality factors
    # Using product to require joint satisfaction, but normalized to avoid collapse
    quality_factor = (velocity_quality * angle_quality * proximity_quality) ** 0.333
    components["contact_landing_reward"] = 5.0 * contact_active * quality_factor
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```
```
