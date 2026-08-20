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

## 4. 语义演化（跨轮组件功能追踪）

按**语义功能**对组件分组，追踪同一功能在多个轮次中的变化。不要用组件名称分组——组件可能在迭代中被重命名，但功能相同。

### 如何识别语义功能

根据组件的**数学形态和奖励方向**判断其功能角色：
- **接近/进度信号**：基于位置/距离的引导，通常为正或 delta 形式（如 goal_approach, progress, proximity, approach_shaping, landing_proximity, goal_distance_progress）
- **着陆/接触信号**：基于腿部接触或着陆事件的奖励/惩罚（如 crash_avoidance, landing_success, contact_reward, touchdown_bonus, stable_landed, safe_contact, contact_bonus, settling_quality, landing_reward）
- **稳定性/约束信号**：基于速度/角度/角速度/推力的惩罚或约束（如 soft_landing, orientation_penalty, velocity_penalty, stability_penalty, fuel_penalty, descent_safety, engine_penalty）
- **完成/成功代理**：条件性的大额正信号或乘积形式的完成接近度（如 success_proxy, completion_bonus）

### 输出格式

```
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆信号 | v1_crash_avoidance → v3_landing_success → v4_contact_reward → v5_landing_success | -214 → -115 → -111 → ? | 稀疏bonus↔连续奖励：2次切换 |
| 稳定性 | v1_soft_landing(死) → v2_soft_landing(全覆盖惩罚) | -214 → -110 | 门控→全高度 |
```

**方向切换**列记录同一语义功能在"稀疏bonus/连续奖励"、"状态值/改善量"、"门控/全高度"等对立设计之间的切换次数。切换 ≥ 2 次意味着反思在该功能上可能正在振荡。只报告客观事实，不给出"应该怎么做"的建议。

```

## User Prompt

```markdown
# 本轮训练反馈 (score=-111.178413)
# Training Feedback

## Final-policy outcome
score=-111.178413, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.079360, -95.165512]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 5.602874 | 95.1% | 98.5% | 100.0% |
| action_cost | -0.067000 | -1.1% | 1.1% | 4.9% |
| angle_penalty | -0.020036 | -0.3% | 0.3% | 100.0% |
| success_landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
    # 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]
    left_contact = obs[6]
    right_contact = obs[7]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 距离势能塑造
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_reward = 5.0 * ((-next_dist) - (-dist))

    # 2. 成功着陆奖励 —— 基于接触转变事件，收紧阈值
    prev_contact = (left_contact > 0.5) or (right_contact > 0.5)
    curr_contact = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landing_event = (
        (not prev_contact) and curr_contact and
        next_dist < 0.3 and
        abs(next_x_vel) < 0.3 and
        abs(next_y_vel) < 0.3 and
        abs(next_body_angle) < 0.3
    )
    success_bonus = 200.0 if landing_event else 0.0

    # 3. 姿态惩罚
    angle_penalty = -0.5 * (body_angle ** 2)

    # 4. 动作代价
    action_cost = -0.02 if action != 0 else 0.0

    total_reward = (
        progress_reward +
        success_bonus +
        angle_penalty +
        action_cost
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }

    return float(total_reward), components
```

# 历史 iter_01 训练反馈
# Training Feedback

## Final-policy outcome
score=-155.975043, len=154.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-180.244147, -114.227696]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 1.582203 | 21.1% | 59.8% | 100.0% |
| action_cost | -1.804000 | -24.1% | 24.1% | 58.4% |
| angle_penalty | -1.140523 | -15.2% | 15.2% | 100.0% |
| landing_velocity_penalty | -0.066830 | -0.9% | 0.9% | 5.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 10/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_01 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Lunar Lander 风格任务奖励函数 v1。

    设计理念:
    - 主学习信号: 基于距离势能的 shaping 奖励，持续引导 agent 靠近目标平台。
    - 软着陆约束: 在下降末段对垂直速度施加 hinge 惩罚，防止高速撞击。
    - 姿态稳定: 轻量二次型角度惩罚，鼓励竖直姿态。
    - 能量效率: 轻量动作代价，抑制不必要的引擎使用。
    未使用稀疏成功/失败奖励，因为没有显式标志且稀疏信号难以在 v1 有效学习。
    """
    # ------------------------------------------------------------
    # 0. 解析观测
    # ------------------------------------------------------------
    # 从 obs 提取状态
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]

    # 从 next_obs 提取状态（用于势能塑造）
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # ------------------------------------------------------------
    # 1. 主学习信号: 距离势能塑造 (goal_progress_guidance)
    # ------------------------------------------------------------
    # 用于持续引导智能体靠近目标 (x=0, y=0)
    # 势能函数: -sqrt(x^2 + y^2)，距离越近势能越高
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    potential_curr = -dist
    potential_next = -next_dist
    
    # 势能塑造: 向目标靠近一步的奖励增量
    progress_reward = 5.0 * (potential_next - potential_curr)
    
    # ------------------------------------------------------------
    # 2. 稳定/安全约束: 软着陆速度惩罚 (soft_landing_safety, 针对垂直速度)
    # ------------------------------------------------------------
    # 在高度较低时（y_pos < 0.5），严格惩罚过大的向下速度
    # 使用 hinge 惩罚：只惩罚超过安全阈值的 y_vel (向下为负，越大越负越危险)
    # 安全阈值：着陆时 y_vel 应 > -0.2 (即下降速度不超过 0.2)
    safe_y_vel_threshold = -0.2
    proximity_gate = max(0.0, 1.0 - y_pos / 0.5)  # y<0.5 时逐渐激活 (0->1)
    proximity_gate = min(1.0, proximity_gate)
    
    # 危险速度部分: 如果 y_vel < -0.2 (下降过快)，则产生惩罚
    dangerous_y_vel = max(0.0, -y_vel + safe_y_vel_threshold)  # 例如 y_vel=-0.8 -> max(0, 0.8-0.2)=0.6
    landing_velocity_penalty = -2.0 * proximity_gate * dangerous_y_vel
    
    # ------------------------------------------------------------
    # 3. 稳定/安全约束: 姿态惩罚 (soft_landing_safety, 针对角度)
    # ------------------------------------------------------------
    # 始终鼓励竖直姿态，但权重适中，不阻碍必要的机动
    # 使用二次型：轻微偏离时惩罚很小，大幅偏离时惩罚变大
    angle_penalty = -0.5 * (body_angle ** 2)
    
    # ------------------------------------------------------------
    # 4. 能量效率: 引擎动作代价 (energy_efficiency)
    # ------------------------------------------------------------
    # 对任何引擎点火（action != 0）施加小惩罚，鼓励利用惯性飞行
    # 注意：权重必须很小，不能压制主任务
    action_cost = -0.02 if action != 0 else 0.0
    
    # ------------------------------------------------------------
    # 5. 总奖励组装
    # ------------------------------------------------------------
    total_reward = (
        progress_reward +
        landing_velocity_penalty +
        angle_penalty +
        action_cost
    )
    
    # 记录各个组件（只记录被加进 total 的）
    components = {
        "progress_shaping": progress_reward,
        "landing_velocity_penalty": landing_velocity_penalty,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }
    
    return float(total_reward), components
```

# 历史 iter_02 训练反馈
# Training Feedback

## Final-policy outcome
score=-119.506414, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.824994, -96.557485]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 5.583480 | 50.5% | 52.3% | 100.0% |
| success_landing_bonus | 5.000000 | 45.2% | 45.2% | 0.1% |
| angle_penalty | -0.204852 | -1.9% | 1.9% | 100.0% |
| action_cost | -0.071000 | -0.6% | 0.6% | 5.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_02 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Lunar Lander 风格任务奖励函数 v2。

    修改:
    - 移除无效的 landing_velocity_penalty，新增 success_landing_bonus:
      当 next_obs 满足软着陆条件时给予 +100 稀疏大奖励，解决 v1 无成功信号的问题。
    - 其余组件保持不变。
    """
    # 0. 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 主学习信号: 距离势能塑造 (goal_progress_guidance)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    potential_curr = -dist
    potential_next = -next_dist
    progress_reward = 5.0 * (potential_next - potential_curr)

    # 2. 成功着陆奖励 (success_landing_bonus)
    # 判定条件: 距离目标极近、速度很小、姿态竖直、至少一个支撑脚触地
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.1 and
        abs(next_x_vel) < 0.2 and
        abs(next_y_vel) < 0.2 and
        abs(next_body_angle) < 0.1 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 姿态惩罚
    angle_penalty = -0.5 * (body_angle ** 2)

    # 4. 动作代价
    action_cost = -0.02 if action != 0 else 0.0

    # 5. 总奖励
    total_reward = (
        progress_reward +
        success_bonus +
        angle_penalty +
        action_cost
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }

    return float(total_reward), components
```

# 历史 iter_03 训练反馈
# Training Feedback

## Final-policy outcome
score=-203.446380, len=154.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-270.266370, -158.776888]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 1.487613 | 17.9% | 52.6% | 100.0% |
| angle_penalty | -2.459905 | -29.7% | 29.7% | 100.0% |
| action_cost | -1.472000 | -17.8% | 17.8% | 47.7% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 11/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_03 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 0. 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 距离势能塑造 (progress_shaping)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    potential_curr = -dist
    potential_next = -next_dist
    progress_reward = 5.0 * (potential_next - potential_curr)

    # 2. 成功着陆奖励 (success_landing_bonus)
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.1 and
        abs(next_x_vel) < 0.2 and
        abs(next_y_vel) < 0.2 and
        abs(next_body_angle) < 0.1 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 坠毁/越界惩罚 (crash_penalty)
    out_of_bounds = abs(next_x_pos) > 1.5
    # 接触地面但未满足软着陆条件，或出界，均视为失败
    crash = (not landed) and (contact_any or out_of_bounds)
    crash_penalty_val = -100.0 if crash else 0.0

    # 4. 姿态惩罚
    angle_penalty_val = -0.5 * (body_angle ** 2)

    # 5. 动作代价
    action_cost_val = -0.02 if action != 0 else 0.0

    # 总奖励
    total_reward = (
        progress_reward +
        success_bonus +
        crash_penalty_val +
        angle_penalty_val +
        action_cost_val
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "crash_penalty": crash_penalty_val,
        "angle_penalty": angle_penalty_val,
        "action_cost": action_cost_val
    }

    return float(total_reward), components
```

# 历史 iter_04 训练反馈
# Training Feedback

## Final-policy outcome
score=-114.375751, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-138.749698, -99.234408]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_landing_bonus | 30.000000 | 83.4% | 83.4% | 0.4% |
| progress_shaping | 5.596924 | 15.6% | 16.1% | 100.0% |
| action_cost | -0.097000 | -0.3% | 0.3% | 7.1% |
| angle_penalty | -0.086575 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_04 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 0. 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 主学习信号: 距离势能塑造 (progress_shaping)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    potential_curr = -dist
    potential_next = -next_dist
    progress_reward = 5.0 * (potential_next - potential_curr)

    # 2. 成功着陆奖励 (success_landing_bonus) —— 放宽阈值
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.2 and
        abs(next_x_vel) < 0.5 and
        abs(next_y_vel) < 0.5 and
        abs(next_body_angle) < 0.2 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 姿态惩罚
    angle_penalty = -0.5 * (body_angle ** 2)

    # 4. 动作代价
    action_cost = -0.02 if action != 0 else 0.0

    # 5. 总奖励
    total_reward = (
        progress_reward +
        success_bonus +
        angle_penalty +
        action_cost
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }

    return float(total_reward), components
```

# 历史 iter_05 训练反馈
# Training Feedback

## Final-policy outcome
score=-46.512046, len=740.150000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-274.011265, 201.880974]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_landing_bonus | 1195.000000 | 97.7% | 97.7% | 1.6% |
| action_cost | -13.469000 | -1.1% | 1.1% | 91.0% |
| progress_shaping | 5.104587 | 0.4% | 0.7% | 99.9% |
| angle_penalty | -6.827954 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_05 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 距离势能塑造
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_reward = 5.0 * ((-next_dist) - (-dist))

    # 2. 成功着陆奖励 —— 进一步放宽阈值
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.4 and
        abs(next_x_vel) < 1.0 and
        abs(next_y_vel) < 1.0 and
        abs(next_body_angle) < 0.5 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 姿态惩罚
    angle_penalty = -0.5 * (body_angle ** 2)

    # 4. 动作代价
    action_cost = -0.02 if action != 0 else 0.0

    total_reward = (
        progress_reward +
        success_bonus +
        angle_penalty +
        action_cost
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }

    return float(total_reward), components
```

# 历史 iter_06 训练反馈
# Training Feedback

## Final-policy outcome
score=-111.178413, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.079360, -95.165512]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 5.602874 | 95.1% | 98.5% | 100.0% |
| action_cost | -0.067000 | -1.1% | 1.1% | 4.9% |
| angle_penalty | -0.020036 | -0.3% | 0.3% | 100.0% |
| success_landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_06 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]
    left_contact = obs[6]
    right_contact = obs[7]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 距离势能塑造
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_reward = 5.0 * ((-next_dist) - (-dist))

    # 2. 成功着陆奖励 —— 基于接触转变事件，收紧阈值
    prev_contact = (left_contact > 0.5) or (right_contact > 0.5)
    curr_contact = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landing_event = (
        (not prev_contact) and curr_contact and
        next_dist < 0.3 and
        abs(next_x_vel) < 0.3 and
        abs(next_y_vel) < 0.3 and
        abs(next_body_angle) < 0.3
    )
    success_bonus = 200.0 if landing_event else 0.0

    # 3. 姿态惩罚
    angle_penalty = -0.5 * (body_angle ** 2)

    # 4. 动作代价
    action_cost = -0.02 if action != 0 else 0.0

    total_reward = (
        progress_reward +
        success_bonus +
        angle_penalty +
        action_cost
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }

    return float(total_reward), components
```

# 修改自传（反思LLM过去每轮的修改决策和结果）
# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `landing_velocity_penalty` → 分数 -156.0 → -119.5（+36.5）✅ 大幅改善
- **你做了什么**：将`landing_velocity_penalty`组件整体替换为`success_landing_bonus`，基于next_obs的接触、位置、速度、角度给出稀疏大正奖励。
- **你为什么这么做**：添加成功奖励后，策略在成功沾盘瞬间获得+100，远大于所有单步惩罚，应能强化接近并软着陆的行为，使最终得分由负转正、terminated中成功终止占比增加。
- **回合长度**：len: 154→68

### v3：修改 `crash_penalty` → 分数 -119.5 → -203.4（-83.9）❌ 明显退步
- **你做了什么**：新增 `crash_penalty` 组件，在接触地面但未满足软着陆条件、或 x 位置超出视区边界时给予 -100，与 success_landing_bonus 互斥，其余组件不变。
- **你为什么这么做**：添加与成功对称的 crash 惩罚后，agent 将被迫避免高速接触和越界，从而逐渐探索减速并在靠近目标时精准控制，最终触发 success_landing_bonus，score 应上升，终止方式可能出现成功终止或 len 增加。
- **回合长度**：len: 68→154

### v4：修改 `crash_penalty` → 分数 -203.4 → -114.4（+89.1）✅ 大幅改善
- **你做了什么**：删除 crash_penalty，将 success_landing_bonus 的判定距离阈值放宽到 0.2，速度阈值放宽到 0.5，角度阈值放宽到 0.2。
- **你为什么这么做**：放宽条件后 success_landing_bonus 激活率将显著上升，为 agent 提供可达的正向终极奖励，驱动减速和软着陆，从而使 score 大幅提升。
- **回合长度**：len: 154→68

### v5：修改 `progress_shaping` → 分数 -114.4 → -46.5（+67.9）✅ 大幅改善
- **你做了什么**：将success_landing_bonus的判定条件中的距离阈值由0.2提至0.4，水平/垂直速度阈值由0.5提至1.0，角度阈值由0.2提至0.5，保持接触条件不变。
- **你为什么这么做**：放宽后触发率应升至5%~15%，每episode平均bonus从30增至50以上，总score有望升至0附近或正数，terminated中成功着陆占比明显增加。
- **回合长度**：len: 68→740

### v6：修改 `success_landing_bonus` → 分数 -46.5 → -111.2（-64.7）❌ 明显退步
- **你做了什么**：将success_landing_bonus修改为仅在从无接触转为有接触时触发，且收紧软着陆条件（距离<0.3，速度绝对值<0.3，角度绝对值<0.3），奖金提高至200。
- **你为什么这么做**：改为一次性事件奖励并收紧阈值后，agent不再有动机在接触后滞留，将趋向成功着陆后自然终止，episode长度应显著下降，terminated比例上升，success_bonus的ep_sum将下降但集中在真正的成功episode，score因此上升。
- **回合长度**：len: 740→68

## 你的修改轨迹（按组件分组）

- **`crash_penalty`**：修改了 **2 次**，分数变化：v3(-84) → v4(+89)
- **`landing_velocity_penalty`**：修改了 **1 次**，分数变化：v2(+36)
- **`progress_shaping`**：修改了 **1 次**，分数变化：v5(+68)
- **`success_landing_bonus`**：修改了 **1 次**，分数变化：v6(-65)

**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_cost + angle_penalty + landing_velocity_penalty + progress_shaping | -155.98 | -155.98 | 0.00 | 154.40 | action_cost=-0.013 angle_penalty=-0.015 landing_velocity_penalty=-0.118 progress_shaping=0.035 | new_best |
| 2 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -119.51 | -119.51 | 0.00 | 68.30 | action_cost=-0.006 angle_penalty=-0.004 progress_shaping=0.080 success_landing_bonus=0.208 | new_best |
| 3 | action_cost + angle_penalty + crash_penalty + progress_shaping + success_landing_bonus | -203.45 | -119.51 | -83.94 | 154.15 | action_cost=-0.015 angle_penalty=-0.034 crash_penalty=-0.425 progress_shaping=-0.007 success_landing_bonus=0.002 | no_meaningful_improvement |
| 4 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -114.38 | -114.38 | 0.00 | 68.45 | action_cost=-0.008 angle_penalty=-0.004 progress_shaping=0.079 success_landing_bonus=0.449 | new_best |
| 5 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -46.51 | -46.51 | 0.00 | 740.15 | action_cost=-0.014 angle_penalty=-0.022 progress_shaping=0.040 success_landing_bonus=28.088 | new_best |
| 6 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -111.18 | -46.51 | -64.67 | 68.45 | action_cost=-0.002 angle_penalty=-0.002 progress_shaping=0.081 success_landing_bonus=0.005 | no_meaningful_improvement |


# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 距离势能塑造
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_reward = 5.0 * ((-next_dist) - (-dist))

    # 2. 成功着陆奖励 —— 进一步放宽阈值
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.4 and
        abs(next_x_vel) < 1.0 and
        abs(next_y_vel) < 1.0 and
        abs(next_body_angle) < 0.5 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 姿态惩罚
    angle_penalty = -0.5 * (body_angle ** 2)

    # 4. 动作代价
    action_cost = -0.02 if action != 0 else 0.0

    total_reward = (
        progress_reward +
        success_bonus +
        angle_penalty +
        action_cost
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }

    return float(total_reward), components
```
```
