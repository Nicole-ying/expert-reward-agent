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
# 本轮训练反馈 (score=227.643558)
# Training Feedback

## Final-policy outcome
score=227.643558, len=374.600000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-11.906847, 281.988416]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 36.599581 | 55.8% | 55.8% | 20.3% |
| descent_incentive | 11.552093 | 17.6% | 17.6% | 87.9% |
| goal_proximity | 7.917188 | 12.1% | 12.5% | 96.4% |
| velocity_penalty | -6.845214 | -10.4% | 10.4% | 97.0% |
| orientation_penalty | -2.401008 | -3.7% | 3.7% | 100.0% |

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
| -32.2 | -150.0 | -4.4 | -2.7 | -9.8 | 246.1 | 157.2 | 223.1 | 248.3 | 236.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 61.025 | 412.079 | 64.461 | 51.543 | 52.380 |
| descent_incentive | 3.736 | 0.154 | 7.226 | 12.638 | 5.816 | 12.711 | 13.873 | 12.321 | 12.771 | 11.499 |
| goal_proximity | 7.432 | -0.781 | 7.855 | 7.945 | 7.894 | 7.950 | 7.939 | 7.938 | 7.948 | 7.923 |
| orientation_penalty | -5.410 | -4.971 | -3.877 | -4.144 | -3.630 | -2.529 | -2.940 | -2.522 | -2.088 | -2.614 |
| velocity_penalty | -9.295 | -2.438 | -3.948 | -4.226 | -3.657 | -6.069 | -5.620 | -6.567 | -6.694 | -7.059 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 9.7% | 33.3% | 20.3% | 9.4% | 27.0% |
| descent_incentive | 64.7% | 38.8% | 86.4% | 87.6% | 93.6% | 90.5% | 71.4% | 81.9% | 90.1% | 85.7% |
| goal_proximity | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% | 96.3% | 98.9% | 97.5% | 96.0% | 96.7% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% | 96.8% | 99.0% | 97.8% | 96.6% | 97.1% |

# 本轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7 reward: L2 — potential-based descent_incentive replaces linear-gate formulation.

    Problem (v6): descent_incentive uses `near_ground * descending` where both terms
    are linear. As y→0, the possible descent per step shrinks (physics), causing the
    gradient to weaken at the most critical moment — the final few cm before touchdown.
    The agent hovers near the pad but cannot bridge the last gap to landing.

    Change:
      - descent_incentive: replaced linear `(1-y/1.5) * (y_cur-y_nxt)` with
        potential-based shaping. Potential φ(y) = 1/(y+0.1). Reward = φ(y_nxt) - φ(y_cur).
        This creates a gradient that naturally INTENSIFIES as y→0, compensating for
        the shrinking descent step size.
      - contact_bonus: weight reduced 4.0→2.0 (supporting change). When contact_bonus
        fires at cp80% (1169 ep_sum), it creates a destabilizing reward spike that
        causes catastrophic forgetting of landing behavior. Lower weight reduces the
        spike while keeping it as a meaningful confirmation signal.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement_delta, squared distance) ---
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2
    delta_y_sq = y_cur ** 2 - y_nxt ** 2
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. descent_incentive (L2 — potential-based shaping) ---
    # Potential φ(y) = 1/(y+eps): increases as y decreases, with derivative
    # dφ/dy = -1/(y+eps)^2 that grows strongly near y=0. This creates a
    # natural funnel: the same descent step yields MORE reward when closer
    # to the ground, exactly where the linear formulation used to weaken.
    #
    # Gated by near_center so agent only gets descent credit when aligned
    # with the landing pad.
    w_desc = 1.5
    eps = 0.1
    potential_cur = 1.0 / (y_cur + eps)
    potential_nxt = 1.0 / (y_nxt + eps)
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)  # active within |x|<0.5
    descent_incentive = w_desc * near_center * max(0.0, potential_nxt - potential_cur)

    # --- 5. contact_bonus (sparse landing confirmation, reduced weight) ---
    # Weight reduced 4.0→2.0 to prevent the reward spike (observed up to 1169
    # ep_sum at cp80%) from destabilizing training and causing catastrophic
    # forgetting of landing behavior.
    w_contact = 2.0
    y_th = 0.1
    x_th = 0.2
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + descent_incentive + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'descent_incentive': descent_incentive,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```

# 历史 iter_01 训练反馈
# Training Feedback

## Final-policy outcome
score=-32.346407, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-63.837338, 10.008682]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 3.758436 | 44.1% | 47.8% | 100.0% |
| orientation_penalty | -3.958825 | -46.5% | 46.5% | 100.0% |
| velocity_penalty | -0.485935 | -5.7% | 5.7% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_01 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward: goal proximity + velocity damping + orientation penalty + safe contact bonus
    
    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement_delta) ---
    # encourage moving toward (0,0) by reducing squared distance
    # larger weight on vertical axis to avoid hard landings
    w_x, w_y = 1.0, 2.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive when approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive when approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (quadratic penalty, gated by distance) ---
    # penalise high speed mainly when close to the pad
    w_v = 0.05
    k_gate = 1.0
    dist_nxt = abs(x_nxt) + abs(y_nxt)   # simple Manhattan gate; could use Euclidean
    gate = 1.0 / (1.0 + k_gate * dist_nxt)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. safe_contact_bonus (joint condition proxy) ---
    # reward when both legs touch and position is near (0,0)
    w_contact = 10.0
    y_th = 0.1   # within 0.1 vertical distance
    x_th = 0.2   # within 0.2 horizontal offset
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt   # 0.0 or 1.0
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return total_reward, components
```

# 历史 iter_02 训练反馈
# Training Feedback

## Final-policy outcome
score=107.129410, len=775.600000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-33.715767, 264.822182]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 154.464574 | 92.5% | 92.5% | 2.8% |
| velocity_penalty | -5.313230 | -3.2% | 3.2% | 99.2% |
| goal_proximity | 3.962801 | 2.4% | 2.5% | 99.1% |
| orientation_penalty | -3.090273 | -1.9% | 1.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_02 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2 reward: goal proximity + height-gated velocity damping + orientation penalty + safe contact bonus

    Key change from v1: velocity penalty now uses exponential height-based gate
    instead of Manhattan distance. Gate is ~1.0 near ground (y≈0) and decays
    rapidly with height, so the agent only needs to slow down when landing.
    Weight increased 10x (0.05 → 0.5) to match the magnitude of other components.
    
    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement_delta) ---
    # encourage moving toward (0,0) by reducing squared distance
    # larger weight on vertical axis to avoid hard landings
    w_x, w_y = 1.0, 2.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive when approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive when approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated, L2 change) ---
    # Strong penalty for high speed only when close to the ground.
    # Gate = exp(-y_nxt / h_scale): ~1.0 at y=0, ~0.37 at y=1, ~0.05 at y=3.
    # This lets the agent fly fast at high altitude but forces deceleration for landing.
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. safe_contact_bonus (joint condition proxy) ---
    # reward when both legs touch and position is near (0,0)
    w_contact = 10.0
    y_th = 0.1   # within 0.1 vertical distance
    x_th = 0.2   # within 0.2 horizontal offset
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt   # 0.0 or 1.0
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```

# 历史 iter_03 训练反馈
# Training Feedback

## Final-policy outcome
score=-26.174304, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-55.963618, 13.343272]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 7.685249 | 45.1% | 48.4% | 100.0% |
| velocity_penalty | -4.758376 | -27.9% | 27.9% | 100.0% |
| orientation_penalty | -4.038467 | -23.7% | 23.7% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_03 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3 reward: rebalanced — reduced contact_bonus dominance, stronger goal_proximity

    Key change from v2: contact_bonus weight reduced 10→4 (was 92.5% magnitude share),
    goal_proximity weights doubled (w_x 1→2, w_y 2→4) to restore continuous guidance.
    This balances the sparse landing bonus against the continuous approach signals.
    
    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement_delta, weights doubled) ---
    # encourage moving toward (0,0) by reducing squared distance
    # larger weight on vertical axis to avoid hard landings
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive when approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive when approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    # Strong penalty for high speed only when close to the ground.
    # Gate = exp(-y_nxt / h_scale): ~1.0 at y=0, ~0.37 at y=1, ~0.05 at y=3.
    # This lets the agent fly fast at high altitude but forces deceleration for landing.
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. safe_contact_bonus (joint condition proxy, weight reduced) ---
    # reward when both legs touch and position is near (0,0)
    # weight reduced from 10.0 to 4.0 to prevent dominance over continuous signals
    w_contact = 4.0
    y_th = 0.1   # within 0.1 vertical distance
    x_th = 0.2   # within 0.2 horizontal offset
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt   # 0.0 or 1.0
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```

# 历史 iter_04 训练反馈
# Training Feedback

## Final-policy outcome
score=183.523455, len=710.250000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[90.971361, 220.088340]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 114.082823 | 86.3% | 86.3% | 4.9% |
| goal_proximity | 7.951091 | 6.0% | 6.2% | 98.1% |
| velocity_penalty | -5.735901 | -4.3% | 4.3% | 98.4% |
| orientation_penalty | -3.386074 | -2.6% | 2.6% | 100.0% |
| descent_incentive | 0.779257 | 0.6% | 0.6% | 72.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_04 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4 reward: added descent_incentive — continuous gradient for final descent phase

    Key change from v3: new `descent_incentive` component rewards downward progress
    when the agent is near the landing zone (y < 0.8, |x| < 0.5). This bridges the
    gap between "hovering near target" (where goal_proximity gradient vanishes at
    low y due to squared-distance form) and "committing to touchdown."
    
    Contact_bonus is kept at w=4.0 as a terminal landing confirmation, now reachable
    thanks to the continuous approach gradient provided by descent_incentive.
    
    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement_delta, squared distance) ---
    # Encourage moving toward (0,0). Larger weight on vertical axis to avoid hard landings.
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive when approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive when approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    # Strong penalty for high speed only when close to the ground.
    # Gate = exp(-y_nxt / h_scale): ~1.0 at y=0, ~0.37 at y=1, ~0.05 at y=3.
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. descent_incentive (NEW — continuous descent gradient near landing zone) ---
    # Reward downward progress when near the landing pad.
    # Gates ensure this only fires when the agent is in position to land:
    #   near_ground: linearly decays from 1 at y=0 to 0 at y>=0.8
    #   near_center: linearly decays from 1 at x=0 to 0 at |x|>=0.5
    #   descending:  positive only when moving downward (y_cur > y_nxt)
    # This provides a continuous gradient that peaks during the final descent,
    # naturally tapering to zero once landed (y≈0, no further descent possible).
    w_desc = 2.0
    near_ground = max(0.0, 1.0 - y_nxt / 0.8)       # active below y=0.8
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)  # active within |x|<0.5
    descending = max(0.0, y_cur - y_nxt)             # positive when going down
    descent_incentive = w_desc * near_ground * near_center * descending

    # --- 5. contact_bonus (sparse landing event bonus) ---
    # Reward when both legs touch and position is near (0,0).
    # Kept at moderate weight — descent_incentive now provides the approach gradient.
    w_contact = 4.0
    y_th = 0.1   # within 0.1 vertical distance
    x_th = 0.2   # within 0.2 horizontal offset
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt   # 0.0 or 1.0
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + descent_incentive + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'descent_incentive': descent_incentive,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```

# 历史 iter_05 训练反馈
# Training Feedback

## Final-policy outcome
score=-15.006998, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-43.536337, 26.845598]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 7.881157 | 40.5% | 43.1% | 100.0% |
| velocity_penalty | -6.826279 | -35.1% | 35.1% | 100.0% |
| orientation_penalty | -3.614935 | -18.6% | 18.6% | 100.0% |
| descent_incentive | 0.624328 | 3.2% | 3.2% | 66.1% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_05 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5 reward: L1 — contact_bonus weight 4.0→5.0 to close the 8% gap to target 200.

    All other components unchanged from v4. The descent_incentive (added in v4)
    successfully enabled the agent to discover landing; now we amplify the
    terminal landing reward to push the mean score above 200.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement_delta, squared distance) ---
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2
    delta_y_sq = y_cur ** 2 - y_nxt ** 2
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. descent_incentive (continuous descent gradient near landing zone) ---
    w_desc = 2.0
    near_ground = max(0.0, 1.0 - y_nxt / 0.8)
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)
    descending = max(0.0, y_cur - y_nxt)
    descent_incentive = w_desc * near_ground * near_center * descending

    # --- 5. contact_bonus (sparse landing event bonus, weight ↑ 4.0→5.0) ---
    w_contact = 5.0
    y_th = 0.1
    x_th = 0.2
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + descent_incentive + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'descent_incentive': descent_incentive,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```

# 历史 iter_06 训练反馈
# Training Feedback

## Final-policy outcome
score=-15.733251, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-41.618148, 19.633448]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 7.860524 | 36.8% | 39.1% | 100.0% |
| velocity_penalty | -5.724752 | -26.8% | 26.8% | 100.0% |
| orientation_penalty | -3.916764 | -18.3% | 18.3% | 100.0% |
| descent_incentive | 3.366429 | 15.8% | 15.8% | 73.8% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_06 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v6 reward: L2 — widened descent_incentive gate + higher weight to end the
    contact_bonus lottery.

    Problem (v4→v5): contact_bonus alternates between dead (0% active) and dominant
    (86-92% mag share). When the agent accidentally discovers landing, it learns;
    when it doesn't, training fails completely. The descent_incentive bridge (added
    in v4) is too weak at w=2.0 and gate y<0.8 to independently guide the agent.

    Changes from v5:
      - descent_incentive: near_ground gate widened 0.8→1.5 (active from higher),
        weight increased 2.0→6.0. This provides ~3-5x stronger continuous descent
        gradient, reducing dependence on sparse contact_bonus.
      - contact_bonus: weight reverted 5.0→4.0 (iter_4 proven value).

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement_delta, squared distance) ---
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2
    delta_y_sq = y_cur ** 2 - y_nxt ** 2
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. descent_incentive (L2 — widened gate + stronger weight) ---
    # Gate widened from 0.8 to 1.5: agent now gets descent gradient from higher
    # altitude, making the signal less dependent on already being near the pad.
    # Weight increased 2.0→6.0: the per-step gradient should now be comparable
    # to goal_proximity (~8 ep_sum), strong enough to guide landing independently.
    w_desc = 6.0
    near_ground = max(0.0, 1.0 - y_nxt / 1.5)       # active below y=1.5 (was 0.8)
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)  # active within |x|<0.5
    descending = max(0.0, y_cur - y_nxt)             # positive when going down
    descent_incentive = w_desc * near_ground * near_center * descending

    # --- 5. contact_bonus (sparse landing event, weight reverted to 4.0) ---
    w_contact = 4.0
    y_th = 0.1
    x_th = 0.2
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + descent_incentive + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'descent_incentive': descent_incentive,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```

# 历史 iter_07 训练反馈
# Training Feedback

## Final-policy outcome
score=227.643558, len=374.600000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-11.906847, 281.988416]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 36.599581 | 55.8% | 55.8% | 20.3% |
| descent_incentive | 11.552093 | 17.6% | 17.6% | 87.9% |
| goal_proximity | 7.917188 | 12.1% | 12.5% | 96.4% |
| velocity_penalty | -6.845214 | -10.4% | 10.4% | 97.0% |
| orientation_penalty | -2.401008 | -3.7% | 3.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_07 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7 reward: L2 — potential-based descent_incentive replaces linear-gate formulation.

    Problem (v6): descent_incentive uses `near_ground * descending` where both terms
    are linear. As y→0, the possible descent per step shrinks (physics), causing the
    gradient to weaken at the most critical moment — the final few cm before touchdown.
    The agent hovers near the pad but cannot bridge the last gap to landing.

    Change:
      - descent_incentive: replaced linear `(1-y/1.5) * (y_cur-y_nxt)` with
        potential-based shaping. Potential φ(y) = 1/(y+0.1). Reward = φ(y_nxt) - φ(y_cur).
        This creates a gradient that naturally INTENSIFIES as y→0, compensating for
        the shrinking descent step size.
      - contact_bonus: weight reduced 4.0→2.0 (supporting change). When contact_bonus
        fires at cp80% (1169 ep_sum), it creates a destabilizing reward spike that
        causes catastrophic forgetting of landing behavior. Lower weight reduces the
        spike while keeping it as a meaningful confirmation signal.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement_delta, squared distance) ---
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2
    delta_y_sq = y_cur ** 2 - y_nxt ** 2
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. descent_incentive (L2 — potential-based shaping) ---
    # Potential φ(y) = 1/(y+eps): increases as y decreases, with derivative
    # dφ/dy = -1/(y+eps)^2 that grows strongly near y=0. This creates a
    # natural funnel: the same descent step yields MORE reward when closer
    # to the ground, exactly where the linear formulation used to weaken.
    #
    # Gated by near_center so agent only gets descent credit when aligned
    # with the landing pad.
    w_desc = 1.5
    eps = 0.1
    potential_cur = 1.0 / (y_cur + eps)
    potential_nxt = 1.0 / (y_nxt + eps)
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)  # active within |x|<0.5
    descent_incentive = w_desc * near_center * max(0.0, potential_nxt - potential_cur)

    # --- 5. contact_bonus (sparse landing confirmation, reduced weight) ---
    # Weight reduced 4.0→2.0 to prevent the reward spike (observed up to 1169
    # ep_sum at cp80%) from destabilizing training and causing catastrophic
    # forgetting of landing behavior.
    w_contact = 2.0
    y_th = 0.1
    x_th = 0.2
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + descent_incentive + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'descent_incentive': descent_incentive,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```

# 修改自传（反思LLM过去每轮的修改决策和结果）
# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `?` → 分数 -32.3 → 107.1（+139.5）✅ 大幅改善
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 1000→776

### v3：修改 `?` → 分数 107.1 → -26.2（-133.3）❌ 明显退步
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 776→1000

### v4：修改 `?` → 分数 -26.2 → 183.5（+209.7）✅ 大幅改善
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 1000→710

### v5：修改 `?` → 分数 183.5 → -15.0（-198.5）❌ 明显退步
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 710→1000

### v6：修改 `?` → 分数 -15.0 → -15.7（-0.7）➖ 小幅退步
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 1000→1000

### v7：修改 `?` → 分数 -15.7 → 227.6（+243.4）✅ 大幅改善
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 1000→375


**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + goal_proximity + orientation_penalty + velocity_penalty | -32.35 | -32.35 | 0.00 | 1000.00 | contact_bonus=2.905 goal_proximity=0.013 orientation_penalty=-0.008 velocity_penalty=-0.005 | new_best |
| 2 | contact_bonus + goal_proximity + orientation_penalty + velocity_penalty | 107.13 | 107.13 | 0.00 | 775.60 | contact_bonus=2.098 goal_proximity=0.010 orientation_penalty=-0.013 velocity_penalty=-0.041 | new_best |
| 3 | contact_bonus + goal_proximity + orientation_penalty + velocity_penalty | -26.17 | 107.13 | -133.30 | 1000.00 | contact_bonus=1.309 goal_proximity=0.018 orientation_penalty=-0.009 velocity_penalty=-0.033 | no_meaningful_improvement |
| 4 | contact_bonus + descent_incentive + goal_proximity + orientation_penalty + velocity_penalty | 183.52 | 183.52 | 0.00 | 710.25 | contact_bonus=1.279 descent_incentive=0.001 goal_proximity=0.019 orientation_penalty=-0.009 velocity_penalty=-0.038 | new_best |
| 5 | contact_bonus + descent_incentive + goal_proximity + orientation_penalty + velocity_penalty | -15.01 | 183.52 | -198.53 | 1000.00 | contact_bonus=1.514 descent_incentive=0.001 goal_proximity=0.017 orientation_penalty=-0.009 velocity_penalty=-0.034 | no_meaningful_improvement |
| 6 | contact_bonus + descent_incentive + goal_proximity + orientation_penalty + velocity_penalty | -15.73 | 183.52 | -199.26 | 1000.00 | contact_bonus=1.275 descent_incentive=0.007 goal_proximity=0.019 orientation_penalty=-0.009 velocity_penalty=-0.036 | no_meaningful_improvement |
| 7 | contact_bonus + descent_incentive + goal_proximity + orientation_penalty + velocity_penalty | 227.64 | 227.64 | 0.00 | 374.60 | contact_bonus=0.673 descent_incentive=0.060 goal_proximity=0.020 orientation_penalty=-0.009 velocity_penalty=-0.042 | target_solved_new_best |

```
