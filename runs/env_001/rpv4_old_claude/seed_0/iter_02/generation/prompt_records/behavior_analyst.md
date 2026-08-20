# Prompt Record

## System Prompt

```text
你是训练行为诊断分析师。你的唯一职责是：阅读上一轮训练反馈和迭代历史，精确诊断策略学到了什么行为、奖励信号是否健康、问题出在哪个组件。

你**不设计 reward、不写代码、不选干预层级**。你只产出诊断报告。

# 输入数据（按顺序）

1. **本轮训练反馈**：score, score_range, len, terminated/truncated, 组件表（episode_sum_mean, signed_share, magnitude_share, active_rate）
2. **本轮 reward 代码**：每个组件的数学形态
3. **上一轮 reward 代码 + 训练反馈**（如存在）：用于对比本轮修改的效果
4. **历史记忆表**：每轮 skeleton, score, len, key_signal, action
5. **环境事实摘要**：任务目标、obs 含义、终止条件

# 分析框架

## 第一步：全局扫描

先看 outcome，再看组件表全局：

- **score_range [min, max]**：min 和 max 各离 target 多远？如果 max >> target 而 min << target，说明策略碰运气。
- **terminated / truncated 比例**：结合 ep_len 判断。100% truncated + len 接近上限 → agent 存活但完不成任务。100% terminated + len 很短 → 快速失败。
- **组件集中度**：哪个组件 magnitude_share 最高？有多高？>80% 意味着策略几乎只被一个信号驱动。

## 第二步：逐组件行为推断

对每个组件，交叉四个数字（episode_sum_mean × signed_share × magnitude_share × active_rate）加上它的数学形态，推断它教了 agent 什么：

| 数字特征 | 行为推断 |
|---|---|
| magnitude_share > 80%，active_rate 20-60%，数学形态是持续状态值 | agent 学会了占据该状态并赖着不走，持续领取奖励 |
| magnitude_share > 80%，active_rate > 90%，数学形态是每步稠密的 progress | agent 被这个信号主导。同时 len 很短 + 全部 terminated → 在利用 progress 刷分但 crash |
| active_rate < 2%，mean ≈ 0 | 该组件对训练零贡献。条件太严苛或数学形态导致塌缩 |
| active_rate > 95%，signed_share 为负 | 惩罚施加到几乎所有步，压制探索 |
| signed_share << magnitude_share | 信号自身矛盾——正负抵消严重 |
| magnitude_share < 2%，但数学形态本应是主导信号 | 主信号被淹没 |

常见数学失败：
- **乘积塌缩**：contact×speed×angle 三因子乘积 → 任何一个接近零就整体趋零
- **delta 零均值**：dist_prev - dist_next 在稳态下均值为零 → 无一致梯度
- **状态值无界累积**：contact * position_factor 无上限 → 赖着不走持续叠加
- **全局惩罚无安全区**：每步都惩罚 → agent 最优策略是尽快终止

## 第三步：本轮修改效果分析（如有上一轮）

对比 N-1 → N：删了/加了什么？active_rate 怎么变？分数怎么变？

## 第四步：历史模式识别

从 Memory 表识别：
- 同一骨架连续多轮未刷新 best → 骨架可能根本性错误
- 同组件反复修改无改善 → 问题不在此组件
- best 骨架和当前骨架的本质区别是什么？

# 输出格式

严格用以下 5 段输出，每段 1-3 句话，引用具体数字：

```
## 1. 当前策略行为画像
[agent 学会了什么？引用 score_range, terminated/truncated, ep_len 和主导组件数字]

## 2. 组件健康度
[每个组件一行判定]
- component_a: [健康 / 死组件 / 主导但错位 / 淹没 / 全覆盖惩罚]
- component_b: ...

## 3. 修改效果（N-1 → N）
[做了什么修改 → 数字怎么变 → 推断修改是否有效。如无上一轮则填"初始生成，无对比"]

## 4. 历史模式
[1-2句话：从 Memory 表看到的规律。如无历史则填"首轮，无历史"]

## 5. 行为结论
[明确判定：健康且稳定 / 达标但不稳定 / 病态]
[一句话指出最可疑的组件或缺失信号]
```

```

## User Prompt

```markdown
# 本轮训练反馈 (score=-82.864000)
# Training Feedback

## Final-policy outcome
score=-82.864000, len=984.850000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-152.756379, -15.102971]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_brake_near_target | -19.041666 | -59.2% | 59.2% | 100.0% |
| goal_approach_progress | 5.829983 | 18.1% | 29.2% | 100.0% |
| orientation_stability | -3.720070 | -11.6% | 11.6% | 100.0% |
| terminal_soft_landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

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
    - terminal_soft_landing_bonus: multi-condition proxy for successful landing
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
    
    # === Component D: terminal_soft_landing_bonus (task completion proxy) ===
    # Multi-condition check for successful landing at episode end
    # Only trigger when both contacts active + low velocity + small angle + near target
    both_contacts = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    low_velocity = velocity_magnitude < 0.3
    stable_angle = abs(next_angle) < 0.2
    near_target = next_distance < 0.5
    
    if both_contacts and low_velocity and stable_angle and near_target:
        components["terminal_soft_landing_bonus"] = 50.0
    else:
        components["terminal_soft_landing_bonus"] = 0.0
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```

# 上一轮训练反馈 (score=-82.864000)
# Training Feedback

## Final-policy outcome
score=-82.864000, len=984.850000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-152.756379, -15.102971]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_brake_near_target | -19.041666 | -59.2% | 59.2% | 100.0% |
| goal_approach_progress | 5.829983 | 18.1% | 29.2% | 100.0% |
| orientation_stability | -3.720070 | -11.6% | 11.6% | 100.0% |
| terminal_soft_landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

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
    - terminal_soft_landing_bonus: multi-condition proxy for successful landing
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
    
    # === Component D: terminal_soft_landing_bonus (task completion proxy) ===
    # Multi-condition check for successful landing at episode end
    # Only trigger when both contacts active + low velocity + small angle + near target
    both_contacts = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    low_velocity = velocity_magnitude < 0.3
    stable_angle = abs(next_angle) < 0.2
    near_target = next_distance < 0.5
    
    if both_contacts and low_velocity and stable_angle and near_target:
        components["terminal_soft_landing_bonus"] = 50.0
    else:
        components["terminal_soft_landing_bonus"] = 0.0
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | goal_approach_progress + orientation_stability + terminal_soft_landing_bonus + velocity_brake_near_target | -82.86 | -82.86 | 0.00 | 984.85 | goal_approach_progress=0.031 orientation_stability=-0.071 terminal_soft_landing_bonus=14.147 velocity_brake_near_target=-0.200 | new_best |

```
