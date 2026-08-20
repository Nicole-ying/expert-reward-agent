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
# 本轮训练反馈 (score=146.319931)
# Training Feedback

## Final-policy outcome
score=146.319931, len=673.350000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[0.897306, 252.418300]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_completion | 47.039200 | 70.5% | 70.5% | 19.0% |
| distance_progress | 5.729716 | 8.6% | 12.9% | 99.5% |
| fuel_penalty | -4.571500 | -6.8% | 6.8% | 67.9% |
| stability | -3.284203 | -4.9% | 4.9% | 100.0% |
| descent_bonus | 3.276273 | 4.9% | 4.9% | 62.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估数据（每10%步数的官方环境得分）
# Checkpoint 评估（训练过程中各组件的变化趋势）

每个 checkpoint 在训练到 N% 时用当前模型评估 20 集，记录每个组件的 ep_sum_mean。
**趋势比单点数值更重要：**
- 组件全程为 0 → 死组件，agent 在训练中忽略了它
- 组件从 0 逐渐增长 → 正在引导学习
- 组件骤升 → 可能被 exploit
- 组件骤降 → agent 在学习抑制它

## Overall score trend
| cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -99.0 | -36.6 | 2.6 | 13.4 | 2.2 | -14.4 | 30.9 | 148.8 | 105.3 | 179.9 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| descent_bonus | 2.621 | 2.519 | 2.662 | 2.738 | 2.942 | 3.121 | 3.180 | 3.340 | 3.412 | 3.224 |
| distance_progress | 5.532 | 4.882 | 5.325 | 5.763 | 6.068 | 6.233 | 6.429 | 5.953 | 5.800 | 5.901 |
| fuel_penalty | -0.131 | -0.576 | -0.644 | -0.638 | -0.683 | -0.697 | -1.034 | -4.935 | -4.856 | -3.868 |
| landing_completion | 0.000 | 0.000 | 0.449 | 0.074 | 0.013 | 0.000 | 0.434 | 24.320 | 24.697 | 24.303 |
| stability | -3.293 | -3.195 | -2.335 | -2.094 | -2.008 | -2.240 | -2.436 | -2.815 | -3.251 | -3.408 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| descent_bonus | 92.9% | 76.0% | 85.5% | 95.6% | 96.3% | 96.5% | 97.6% | 65.3% | 75.5% | 71.8% |
| distance_progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.4% | 99.5% | 98.9% |
| fuel_penalty | 18.9% | 56.1% | 59.1% | 59.2% | 58.0% | 56.5% | 63.6% | 68.7% | 65.2% | 66.2% |
| landing_completion | 0.0% | 0.0% | 2.6% | 1.3% | 0.6% | 0.0% | 2.0% | 9.9% | 12.1% | 11.9% |
| stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 本轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lunar lander-like navigation task.
    Components:
      A. distance_progress   – reduction in Euclidean distance to landing pad
      B. stability           – penalise high velocity and large attitude when close to pad
      C. descent_bonus       – reward vertical descent when horizontally aligned with pad
      D. landing_completion  – small soft success bonus for two-leg contact with low speed
      E. fuel_penalty        – tiny penalty for using any engine
    """
    # --- unpack observations ---
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx_pos = next_obs[0]
    ny_pos = next_obs[1]

    # --- distances ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    velocity_norm = (x_vel**2 + y_vel**2) ** 0.5

    # =====================================================================
    # A. distance_progress: reward reduction in Euclidean distance to target
    # =====================================================================
    progress = (current_distance - next_distance) * 5.0   # scaling factor

    # =====================================================================
    # B. stability: smooth penalty for speed and attitude when near the pad
    #    gate: exp(-2 * distance)  →  close to 1 when distance small
    # =====================================================================
    gate = 2.718281828 ** (-2.0 * current_distance)     # approximate exp(-2d)
    w_vel = 0.1
    w_angle = 0.5
    w_angv = 0.5
    stability_penalty = - (w_vel  * velocity_norm**2 +
                           w_angle * angle**2 +
                           w_angv  * ang_vel**2) * gate

    # =====================================================================
    # C. descent_bonus: reward vertical descent when horizontally aligned
    #    with the landing pad.  This provides continuous gradient from
    #    altitude to ground, bridging the gap between distance_progress
    #    (which is isotropic) and landing_completion (which is sparse).
    #
    #    Key design: rewards the *change* in y (delta-y), NOT the absolute
    #    y position.  Hovering at low altitude gives zero reward → no exploit.
    #    Gate ensures reward only activates when agent is above the pad.
    # =====================================================================
    x_gate = max(0.0, 1.0 - abs(x_pos) / 0.5)          # 1 at x=0, 0 at |x|>=0.5
    descent = max(0.0, y_pos - ny_pos)                  # positive when descending
    descent_bonus = 3.0 * x_gate * descent

    # =====================================================================
    # D. landing_completion: small soft success bonus – both legs on ground,
    #    near centre, with low velocity and stable attitude.
    #    Reduced weight (0.5 instead of 2.0) to avoid reward-spike instability.
    # =====================================================================
    both_feet = left_contact * right_contact            # 1.0 only if both are 1.0
    close_to_centre = current_distance < 0.3
    if both_feet > 0.5 and close_to_centre:
        speed_quality = max(0.0, 1.0 - velocity_norm / 0.5)
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.5)
        angvel_quality = max(0.0, 1.0 - abs(ang_vel) / 0.5)
        landing_reward = 0.5 * speed_quality * angle_quality * angvel_quality
    else:
        landing_reward = 0.0

    # =====================================================================
    # E. fuel_penalty: minuscule cost for any engine use
    # =====================================================================
    fuel_penalty = -0.01 if action != 0 else 0.0

    total_reward = (progress + stability_penalty + descent_bonus
                    + landing_reward + fuel_penalty)

    components = {
        "distance_progress": progress,
        "stability": stability_penalty,
        "descent_bonus": descent_bonus,
        "landing_completion": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```

# 历史 iter_01 训练反馈
# Training Feedback

## Final-policy outcome
score=-203.530758, len=185.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-298.518231, -157.142875]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -0.668828 | -44.8% | 44.8% | 15.5% |
| angular_vel_penalty | -0.433896 | -29.1% | 29.1% | 100.0% |
| progress | -0.199657 | -13.4% | 26.2% | 100.0% |
| landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 10/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_01 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步观测
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # 下一步观测
    next_x, next_y, next_vx, next_vy, next_angle, next_ang_vel, next_left_contact, next_right_contact = next_obs

    # 1. 主进展：鼓励向目标垫 (0,0) 靠近（距离减小）
    old_dist = (x ** 2 + y ** 2) ** 0.5
    new_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = old_dist - new_dist          # >0 接近，<0 远离
    w_dist = 1.0
    progress_reward = w_dist * progress

    # 2. 靠近目标时的速度惩罚：强制减速，防止高速撞击
    closeness = max(0.0, 1.0 - new_dist / 1.5)    # 距离<1.5时激活，距离0时=1
    speed = abs(next_vx) + abs(next_vy)
    w_speed = 0.5
    speed_penalty = -w_speed * closeness * speed

    # 3. 角速度轻量惩罚：保持姿态稳定
    w_angvel = 0.1
    angular_vel_penalty = -w_angvel * abs(next_ang_vel)

    # 4. 着陆奖励：双脚接触 + 位置对中 + 低速 + 小角度
    x_thresh = 0.3
    y_thresh = 0.3
    vel_thresh = 0.5
    angle_thresh = 0.2  # rad ≈ 11°

    contact_factor = next_left_contact * next_right_contact  # 0 或 1
    pos_x_factor = max(0.0, 1.0 - abs(next_x) / x_thresh)
    pos_y_factor = max(0.0, 1.0 - abs(next_y) / y_thresh)
    vel_factor = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / vel_thresh)
    angle_factor = max(0.0, 1.0 - abs(next_angle) / angle_thresh)

    landing_multiplier = (contact_factor * pos_x_factor * pos_y_factor
                          * vel_factor * angle_factor)
    w_land = 2.0
    landing_reward = w_land * landing_multiplier

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + landing_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing": landing_reward
    }
    return float(total_reward), components
```

# 历史 iter_02 训练反馈
# Training Feedback

## Final-policy outcome
score=-37.031320, len=962.550000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-119.359794, 188.512930]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing | 697.436041 | 93.8% | 93.8% | 99.6% |
| speed_penalty | -42.129938 | -5.7% | 5.7% | 100.0% |
| angular_vel_penalty | -2.456624 | -0.3% | 0.3% | 100.0% |
| progress | 0.664903 | 0.1% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_02 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步观测
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # 下一步观测
    next_x, next_y, next_vx, next_vy, next_angle, next_ang_vel, next_left_contact, next_right_contact = next_obs

    # 1. 主进展：鼓励向目标垫 (0,0) 靠近（距离减小）
    old_dist = (x ** 2 + y ** 2) ** 0.5
    new_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = old_dist - new_dist          # >0 接近，<0 远离
    w_dist = 1.0
    progress_reward = w_dist * progress

    # 2. 靠近目标时的速度惩罚：强制减速，防止高速撞击
    closeness = max(0.0, 1.0 - new_dist / 1.5)    # 距离<1.5时激活，距离0时=1
    speed = abs(next_vx) + abs(next_vy)
    w_speed = 0.5
    speed_penalty = -w_speed * closeness * speed

    # 3. 角速度轻量惩罚：保持姿态稳定
    w_angvel = 0.1
    angular_vel_penalty = -w_angvel * abs(next_ang_vel)

    # 4. 着陆奖励：proximity门控 + sum-of-factors，提供连续梯度
    # 激活条件：进入目标附近（曼哈顿距离 < 1.5）
    proximity = max(0.0, 1.0 - (abs(next_x) + abs(next_y)) / 1.5)

    if proximity > 0.0:
        # 各项着陆质量因子（放松阈值，每个在 [0,1]）
        pos_x_factor = max(0.0, 1.0 - abs(next_x) / 0.5)
        pos_y_factor = max(0.0, 1.0 - abs(next_y) / 0.5)
        vel_factor = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / 1.0)
        angle_factor = max(0.0, 1.0 - abs(next_angle) / 0.5)
        # 接触：0/0.5/1.0，作为 soft bonus
        contact_score = (next_left_contact + next_right_contact) / 2.0

        # sum-of-factors 取平均：任何方面的改善都有梯度
        landing_score = proximity * (
            pos_x_factor + pos_y_factor + vel_factor + angle_factor + contact_score
        ) / 5.0
    else:
        landing_score = 0.0

    w_land = 3.0
    landing_reward = w_land * landing_score

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + landing_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing": landing_reward
    }
    return float(total_reward), components
```

# 历史 iter_03 训练反馈
# Training Feedback

## Final-policy outcome
score=-222.850054, len=209.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-336.317345, -168.881892]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -0.735491 | -46.9% | 46.9% | 14.8% |
| progress | -0.298984 | -19.1% | 29.6% | 100.0% |
| angular_vel_penalty | -0.367962 | -23.5% | 23.5% | 100.0% |
| landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 7/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_03 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步观测
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # 下一步观测
    next_x, next_y, next_vx, next_vy, next_angle, next_ang_vel, next_left_contact, next_right_contact = next_obs

    # 1. 主进展：鼓励向目标垫 (0,0) 靠近（距离减小）
    old_dist = (x ** 2 + y ** 2) ** 0.5
    new_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = old_dist - new_dist          # >0 接近，<0 远离
    w_dist = 1.0
    progress_reward = w_dist * progress

    # 2. 靠近目标时的速度惩罚：强制减速，防止高速撞击
    closeness = max(0.0, 1.0 - new_dist / 1.5)    # 距离<1.5时激活，距离0时=1
    speed = abs(next_vx) + abs(next_vy)
    w_speed = 0.5
    speed_penalty = -w_speed * closeness * speed

    # 3. 角速度轻量惩罚：保持姿态稳定
    w_angvel = 0.1
    angular_vel_penalty = -w_angvel * abs(next_ang_vel)

    # 4. 着陆奖励：contact-gated + quality-scaled
    #    仅在脚接触目标垫时激活，杜绝悬停 exploit
    contact = (next_left_contact + next_right_contact) / 2.0  # 0 / 0.5 / 1.0

    if contact > 0.0:
        # 各项着陆质量因子（每个在 [0,1]）
        pos_quality = max(0.0, 1.0 - (abs(next_x) + abs(next_y)) / 0.5)
        vel_quality = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / 1.0)
        angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)

        # contact 作为放大器：单脚=0.5x，双脚=1.0x
        # 任何质量因子的改善都有连续梯度
        landing_score = contact * (pos_quality + vel_quality + angle_quality) / 3.0
    else:
        landing_score = 0.0

    w_land = 5.0  # 提升权重补偿稀疏性
    landing_reward = w_land * landing_score

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + landing_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing": landing_reward
    }
    return float(total_reward), components
```

# 历史 iter_04 训练反馈
# Training Feedback

## Final-policy outcome
score=-295.326995, len=138.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-343.965185, -239.829467]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_guidance | -1.595910 | -18.9% | 53.9% | 100.0% |
| speed_penalty | -2.096809 | -24.9% | 24.9% | 63.2% |
| angular_vel_penalty | -1.265816 | -15.0% | 15.0% | 100.0% |
| progress | -0.026234 | -0.3% | 6.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 11/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_04 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步观测
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # 下一步观测
    next_x, next_y, next_vx, next_vy, next_angle, next_ang_vel, next_left_contact, next_right_contact = next_obs

    # 1. 主进展：鼓励向目标垫 (0,0) 靠近（距离减小）
    old_dist = (x ** 2 + y ** 2) ** 0.5
    new_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = old_dist - new_dist          # >0 接近，<0 远离
    w_dist = 1.0
    progress_reward = w_dist * progress

    # 2. 靠近目标时的速度惩罚：强制减速，防止高速撞击
    closeness = max(0.0, 1.0 - new_dist / 1.5)    # 距离<1.5时激活，距离0时=1
    speed = abs(next_vx) + abs(next_vy)
    w_speed = 0.5
    speed_penalty = -w_speed * closeness * speed

    # 3. 角速度轻量惩罚：保持姿态稳定
    w_angvel = 0.1
    angular_vel_penalty = -w_angvel * abs(next_ang_vel)

    # 4. 速度方向引导（新组件，替代 landing）
    #    奖励「速度方向指向目标」，提供即时正向导航梯度
    #    距离衰减：远处需要导航引导，近处让减速信号接管
    if old_dist > 0.05:
        # 单位方向向量：从当前位置指向目标 (0,0)
        dir_x = -x / old_dist
        dir_y = -y / old_dist

        # 速度方向与目标方向的余弦相似度
        spd = (next_vx ** 2 + next_vy ** 2) ** 0.5 + 0.01
        alignment = (next_vx * dir_x + next_vy * dir_y) / spd  # [-1, 1]

        # 距离衰减权重：远处=1.0，距离0→0（线性衰减，5单位外满权重）
        distance_weight = min(1.0, old_dist / 5.0)

        w_guidance = 0.5
        guidance_reward = w_guidance * distance_weight * alignment
    else:
        guidance_reward = 0.0

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + guidance_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "velocity_guidance": guidance_reward
    }
    return float(total_reward), components
```

# 历史 iter_05 训练反馈
# Training Feedback

## Final-policy outcome
score=-197.455359, len=185.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-292.101504, -148.214023]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -0.716674 | -49.3% | 49.3% | 15.8% |
| progress | -0.182647 | -12.6% | 27.1% | 100.0% |
| angular_vel_penalty | -0.342899 | -23.6% | 23.6% | 100.0% |
| landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 7/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_05 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步观测
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # 下一步观测
    next_x, next_y, next_vx, next_vy, next_angle, next_ang_vel, next_left_contact, next_right_contact = next_obs

    # 1. 主进展：鼓励向目标垫 (0,0) 靠近（距离减小）
    old_dist = (x ** 2 + y ** 2) ** 0.5
    new_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = old_dist - new_dist          # >0 接近，<0 远离
    w_dist = 1.0
    progress_reward = w_dist * progress

    # 2. 靠近目标时的速度惩罚：强制减速，防止高速撞击
    closeness = max(0.0, 1.0 - new_dist / 1.5)    # 距离<1.5时激活，距离0时=1
    speed = abs(next_vx) + abs(next_vy)
    w_speed = 0.5
    speed_penalty = -w_speed * closeness * speed

    # 3. 角速度轻量惩罚：保持姿态稳定
    w_angvel = 0.1
    angular_vel_penalty = -w_angvel * abs(next_ang_vel)

    # 4. 着陆奖励：proximity门控 + contact放大
    #    设计要点：
    #    - proximity 门控保证可发现性（进入目标附近即有信号）
    #    - contact 放大提供着陆激励（双脚接触时 5x 奖励）
    #    - 低权重防止悬停 exploit（无接触时 max=0.5/步，远低于 iter2 的 3.0/步）
    proximity = max(0.0, 1.0 - (abs(next_x) + abs(next_y)) / 1.0)  # Manhattan < 1.0

    if proximity > 0.0:
        # 着陆质量因子（每个在 [0,1]）
        pos_quality = max(0.0, 1.0 - (abs(next_x) + abs(next_y)) / 0.3)
        vel_quality = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / 0.5)
        angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.3)

        # 接触信号：0 / 0.5 / 1.0
        contact = (next_left_contact + next_right_contact) / 2.0

        # contact 放大倍数：无接触=1x，单脚=3x，双脚=5x
        contact_mult = 1.0 + 4.0 * contact

        # 最终着陆分数：proximity × contact_mult × 平均质量
        landing_score = proximity * contact_mult * (
            pos_quality + vel_quality + angle_quality
        ) / 3.0
    else:
        landing_score = 0.0

    w_land = 0.5
    landing_reward = w_land * landing_score

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + landing_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing": landing_reward
    }
    return float(total_reward), components
```

# 历史 iter_06 训练反馈
# Training Feedback

## Final-policy outcome
score=-13.573963, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-45.504561, 26.677289]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| fuel_penalty | -9.888500 | -53.2% | 53.2% | 98.9% |
| distance_progress | 6.331350 | 34.1% | 39.1% | 100.0% |
| stability | -1.416178 | -7.6% | 7.6% | 100.0% |
| landing_completion | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_06 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lunar lander-like navigation task.
    Components:
      A. distance_progress   – reduction in Euclidean distance to landing pad
      B. stability           – penalise high velocity and large attitude when close to pad
      C. descent_bonus       – reward vertical descent when horizontally aligned with pad
      D. landing_completion  – small soft success bonus for two-leg contact with low speed
      E. fuel_penalty        – tiny penalty for using any engine
    """
    # --- unpack observations ---
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx_pos = next_obs[0]
    ny_pos = next_obs[1]

    # --- distances ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    velocity_norm = (x_vel**2 + y_vel**2) ** 0.5

    # =====================================================================
    # A. distance_progress: reward reduction in Euclidean distance to target
    # =====================================================================
    progress = (current_distance - next_distance) * 5.0   # scaling factor

    # =====================================================================
    # B. stability: smooth penalty for speed and attitude when near the pad
    #    gate: exp(-2 * distance)  →  close to 1 when distance small
    # =====================================================================
    gate = 2.718281828 ** (-2.0 * current_distance)     # approximate exp(-2d)
    w_vel = 0.1
    w_angle = 0.5
    w_angv = 0.5
    stability_penalty = - (w_vel  * velocity_norm**2 +
                           w_angle * angle**2 +
                           w_angv  * ang_vel**2) * gate

    # =====================================================================
    # C. descent_bonus: reward vertical descent when horizontally aligned
    #    with the landing pad.  This provides continuous gradient from
    #    altitude to ground, bridging the gap between distance_progress
    #    (which is isotropic) and landing_completion (which is sparse).
    #
    #    Key design: rewards the *change* in y (delta-y), NOT the absolute
    #    y position.  Hovering at low altitude gives zero reward → no exploit.
    #    Gate ensures reward only activates when agent is above the pad.
    # =====================================================================
    x_gate = max(0.0, 1.0 - abs(x_pos) / 0.5)          # 1 at x=0, 0 at |x|>=0.5
    descent = max(0.0, y_pos - ny_pos)                  # positive when descending
    descent_bonus = 3.0 * x_gate * descent

    # =====================================================================
    # D. landing_completion: small soft success bonus – both legs on ground,
    #    near centre, with low velocity and stable attitude.
    #    Reduced weight (0.5 instead of 2.0) to avoid reward-spike instability.
    # =====================================================================
    both_feet = left_contact * right_contact            # 1.0 only if both are 1.0
    close_to_centre = current_distance < 0.3
    if both_feet > 0.5 and close_to_centre:
        speed_quality = max(0.0, 1.0 - velocity_norm / 0.5)
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.5)
        angvel_quality = max(0.0, 1.0 - abs(ang_vel) / 0.5)
        landing_reward = 0.5 * speed_quality * angle_quality * angvel_quality
    else:
        landing_reward = 0.0

    # =====================================================================
    # E. fuel_penalty: minuscule cost for any engine use
    # =====================================================================
    fuel_penalty = -0.01 if action != 0 else 0.0

    total_reward = (progress + stability_penalty + descent_bonus
                    + landing_reward + fuel_penalty)

    components = {
        "distance_progress": progress,
        "stability": stability_penalty,
        "descent_bonus": descent_bonus,
        "landing_completion": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```

# 修改自传（反思LLM过去每轮的修改决策和结果）
# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `?` → 分数 -203.5 → -37.0（+166.5）✅ 大幅改善
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 186→963

### v3：修改 `?` → 分数 -37.0 → -222.9（-185.8）❌ 明显退步
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 963→210

### v4：修改 `velocity_guidance` → 分数 -222.9 → -295.3（-72.5）❌ 明显退步
- **你做了什么**：骨架重建：组件从 ['angular_vel_penalty', 'landing', 'progress', 'speed_penalty'] 变为 ['angular_vel_penalty', 'progress', 'speed_p...
- **回合长度**：len: 210→139

### v5：修改 `velocity_guidance` → 分数 -295.3 → -197.5（+97.9）✅ 大幅改善
- **你做了什么**：骨架重建：组件从 ['angular_vel_penalty', 'progress', 'speed_penalty', 'velocity_guidance'] 变为 ['angular_vel_penalty', 'landing',...
- **回合长度**：len: 139→186

### v7：修改 `?` → 分数 -13.6 → 146.3（+159.9）✅ 大幅改善
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 1000→673


**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angular_vel_penalty + landing + progress + speed_penalty | -203.53 | -203.53 | 0.00 | 185.80 | angular_vel_penalty=-0.006 landing=0.001 progress=-0.005 speed_penalty=-0.012 | new_best |
| 2 | angular_vel_penalty + landing + progress + speed_penalty | -37.03 | -37.03 | 0.00 | 962.55 | angular_vel_penalty=-0.008 landing=1.538 progress=0.003 speed_penalty=-0.074 | new_best |
| 3 | angular_vel_penalty + landing + progress + speed_penalty | -222.85 | -37.03 | -185.82 | 209.60 | angular_vel_penalty=-0.005 landing=0.003 progress=-0.004 speed_penalty=-0.012 | no_meaningful_improvement |
| 4 | angular_vel_penalty + progress + speed_penalty + velocity_guidance | -295.33 | -37.03 | -258.30 | 138.90 | angular_vel_penalty=-0.019 progress=-0.001 speed_penalty=-0.040 velocity_guidance=-0.016 | no_meaningful_improvement |
| 5 | angular_vel_penalty + landing + progress + speed_penalty | -197.46 | -37.03 | -160.42 | 185.50 | angular_vel_penalty=-0.005 landing=0.003 progress=-0.004 speed_penalty=-0.015 | unsolved_stagnation_fresh_restart |
| 6 | distance_progress + fuel_penalty + landing_completion + stability | -13.57 | -13.57 | 0.00 | 1000.00 | distance_progress=0.016 fuel_penalty=-0.007 landing_completion=0.864 stability=-0.009 | new_best |
| 7 | descent_bonus + distance_progress + fuel_penalty + landing_completion + stability | 146.32 | 146.32 | 0.00 | 673.35 | descent_bonus=0.024 distance_progress=0.048 fuel_penalty=-0.006 landing_completion=0.016 stability=-0.023 | new_best |

```
