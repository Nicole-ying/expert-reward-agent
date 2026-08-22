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
# 本轮训练反馈 (score=-5.371040)
# Training Feedback

## Final-policy outcome
score=-5.371040, len=964.100000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-37.696217, 118.125687]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 95.903507 | 86.1% | 86.1% | 0.7% |
| upright_penalty | -5.454539 | -4.9% | 4.9% | 100.0% |
| goal_proximity | 4.753808 | 4.3% | 4.8% | 100.0% |
| velocity_penalty | -4.620901 | -4.1% | 4.1% | 100.0% |

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
| -116.2 | 2.0 | 118.6 | 52.9 | 5.8 | -67.4 | 29.4 | 111.8 | -40.5 | -13.5 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 18.192 | 379.679 | 2054.686 | 128.376 | 254.282 | 107.466 | 239.389 | 454.420 | 0.000 | 4.608 |
| goal_proximity | 4.499 | 3.389 | 4.231 | 3.558 | 3.221 | 2.657 | 4.056 | 3.855 | 3.884 | 4.766 |
| upright_penalty | -8.336 | -16.783 | -7.820 | -10.222 | -5.381 | -4.245 | -8.311 | -10.238 | -3.689 | -3.602 |
| velocity_penalty | -20.782 | -11.080 | -8.380 | -10.302 | -6.472 | -6.126 | -5.228 | -5.419 | -4.928 | -4.536 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 1.9% | 11.5% | 25.2% | 5.3% | 2.7% | 0.8% | 4.1% | 10.1% | 0.0% | 0.0% |
| goal_proximity | 100.0% | 100.0% | 99.9% | 99.2% | 99.7% | 99.8% | 99.2% | 98.6% | 100.0% | 100.0% |
| upright_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 99.9% | 100.0% | 99.8% | 99.1% | 99.7% | 99.8% | 99.2% | 98.6% | 100.0% | 100.0% |

# 本轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 7): landing_approach replaced with contact_bonus.
    The environment card identifies contact_success as a mandatory role —
    the only explicit "landing complete" signal. Iter 6's landing_approach
    was a continuous descent potential that never peaked at the landing event,
    providing only 0.974 ep_sum with zero successful landings (0/20).

    contact_bonus = 20 * centering_factor (moderate, between the dominating
    50 of iter 2 and the dead 8 of iter 4-5). No velocity gate — the
    velocity_penalty already enforces soft landing, and iter 3's velocity
    gate (exp(-3*speed²)) caused score collapse from 64→21 by making the
    bonus too hard to trigger.

    Velocity penalty kept at 0.3 (softened in iter 6 from 0.5) to avoid
    the "penalty wall" that prevented the agent from approaching the pad.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.3 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: moderate terminal reward for dual-leg landing ---
    # Restores the mandatory contact_success role (removed in iter 6).
    # Value 20 is chosen between the dominating 50 (iter 2) and dead 8 (iter 4-5).
    # centering_factor ensures off-pad landings receive proportionally less.
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 20.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return float(total_reward), components
```

# 历史 iter_01 训练反馈
# Training Feedback

## Final-policy outcome
score=-76.561074, len=975.150000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-109.556126, -13.998983]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -12.561269 | -71.7% | 71.7% | 100.0% |
| upright_penalty | -3.714416 | -21.2% | 21.2% | 100.0% |
| goal_proximity | 0.635053 | 3.6% | 7.2% | 100.0% |
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
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.
    """
    # --- goal proximity: encourage reducing Euclidean distance to pad ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 1.0 * (prev_dist - next_dist)   # delta reward

    # --- soft landing velocity penalty: quadratic penalty on velocity components ---
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2)

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: reward simultaneous leg contact, modulated by horizontal centering ---
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian-like decay with horizontal offset
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 50.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```

# 历史 iter_02 训练反馈
# Training Feedback

## Final-policy outcome
score=63.720709, len=668.050000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-208.175480, 265.833071]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 1075.630017 | 98.2% | 98.2% | 5.2% |
| velocity_penalty | -9.516978 | -0.9% | 0.9% | 99.0% |
| upright_penalty | -7.300626 | -0.7% | 0.7% | 100.0% |
| goal_proximity | 1.835102 | 0.2% | 0.3% | 98.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_02 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 2): velocity penalty is now height-gated via
    exp(-height / 2.0). At high altitude, velocity penalty is relaxed,
    allowing the agent to move freely toward the target. Near the ground,
    the penalty ramps up to full strength for soft landing.
    Goal proximity weight increased to 2.0 to strengthen approach incentive.
    """
    # --- goal proximity: encourage reducing Euclidean distance to pad ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 2.0 * (prev_dist - next_dist)   # delta reward (weight raised from 1.0)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)  # clamp to avoid amplification below pad
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: reward simultaneous leg contact, modulated by horizontal centering ---
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian-like decay with horizontal offset
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 50.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```

# 历史 iter_03 训练反馈
# Training Feedback

## Final-policy outcome
score=21.452647, len=864.550000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-121.813880, 228.013400]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 1011.467861 | 97.7% | 97.7% | 8.2% |
| upright_penalty | -11.451934 | -1.1% | 1.1% | 100.0% |
| velocity_penalty | -9.103126 | -0.9% | 0.9% | 99.9% |
| goal_proximity | 1.768764 | 0.2% | 0.3% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_03 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 3): contact_bonus now velocity-gated via
    exp(-3.0 * speed_sq). Only low-velocity contacts receive the full bonus;
    high-speed touchdowns are strongly penalized by near-zero bonus.
    This forces the agent to decelerate before ground contact rather than
    relying on the sparse bonus to mask hard landings.
    """
    # --- goal proximity: encourage reducing Euclidean distance to pad ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 2.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: velocity-gated soft landing reward ---
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian decay with horizontal offset from pad center
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        # Velocity quality gate: only reward near-zero-speed contacts
        speed_sq = next_obs[2]**2 + next_obs[3]**2
        velocity_quality = 2.718281828 ** (-3.0 * speed_sq)
        contact_bonus = 50.0 * centering_factor * velocity_quality

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```

# 历史 iter_04 训练反馈
# Training Feedback

## Final-policy outcome
score=-10.108862, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-43.149389, 24.819673]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -7.837401 | -47.6% | 47.6% | 100.0% |
| upright_penalty | -5.113069 | -31.0% | 31.0% | 100.0% |
| goal_proximity | 2.922025 | 17.7% | 21.4% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_04 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 4): goal_proximity switched from linear delta-distance
    to inverse-distance potential — provides much stronger per-step gradients,
    especially near the ground where fine deceleration control matters.
    contact_bonus magnitude reduced from 50→8 and velocity gating removed
    (gating created a chicken-and-egg problem in iter 3, dropping score from
    64→21). The lower bonus magnitude prevents it from drowning out the new
    dense progress signal.
    """
    # --- goal proximity: inverse-distance potential for dense gradients ---
    # phi(d) = 1/(d + epsilon)  ->  reward = phi(next) - phi(prev)
    # This form amplifies gradients near the target (where soft landing
    # requires precise control) while still providing directional signal
    # at moderate distances.
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    epsilon = 0.5  # prevents singularity at d=0, controls gradient sharpness
    prev_potential = 1.0 / (prev_dist + epsilon)
    next_potential = 1.0 / (next_dist + epsilon)
    goal_proximity = 3.0 * (next_potential - prev_potential)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: moderate reward for dual-leg contact ---
    # Velocity gating removed (iter 3 regression evidence).
    # Magnitude reduced 50→8 to prevent drowning dense progress signal.
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian decay with horizontal offset from pad center
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 8.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```

# 历史 iter_05 训练反馈
# Training Feedback

## Final-policy outcome
score=-33.040122, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-67.620306, 12.216166]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -10.121546 | -50.6% | 50.6% | 100.0% |
| goal_proximity | 4.285993 | 21.4% | 31.4% | 100.0% |
| upright_penalty | -3.598494 | -18.0% | 18.0% | 100.0% |
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
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 5): goal_proximity reverted from inverse-distance delta
    back to linear distance delta. Inverse-distance potential (iter 4) caused
    catastrophic training collapse: negligible gradients at moderate distances
    meant the agent received almost no approach signal, learning to hover
    instead. Linear delta provides uniform per-step gradient at all distances,
    which was proven to work in iter 2 (score 63.72, 17/20 terminated).
    Weight raised to 4.0 (from iter 2's 2.0) to compensate for the smaller
    contact_bonus (8 vs 50) and ensure net continuous signals are positive.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: moderate reward for dual-leg contact ---
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian decay with horizontal offset from pad center
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 8.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```

# 历史 iter_06 训练反馈
# Training Feedback

## Final-policy outcome
score=-81.797020, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-149.216081, 84.983972]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| upright_penalty | -3.075439 | -29.4% | 29.4% | 100.0% |
| goal_proximity | 1.774610 | 17.0% | 25.7% | 100.0% |
| velocity_penalty | -2.385383 | -22.8% | 22.8% | 100.0% |
| landing_approach | 0.974175 | 9.3% | 22.1% | 98.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_06 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 6): contact_bonus replaced with landing_approach —
    a continuous potential-based reward that provides gradient all the way
    to the surface. Unlike the sparse binary contact_bonus (which oscillated
    between dead and 98% magnitude-share across iters 1-5), this signal:
      - rewards descending when horizontally centered (x≈0)
      - suppresses descent reward when far from center (via exp(-5x²))
      - provides uniform per-step gradient at all altitudes (no dead zones)
      - cannot explode: total per-episode reward is bounded by the potential
        difference between start and goal states.

    Velocity penalty weight reduced 0.5→0.3 to soften the "near-ground
    penalty wall" and allow the new landing_approach signal to effectively
    guide the final descent phase.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.3 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- landing approach: continuous surface-proximity potential ---
    # phi(x, y) = -y * exp(-alpha * x²)
    #   High potential when low (small y) AND centered (small |x|).
    #   The exp(-5x²) centering gate suppresses descent reward when the
    #   agent is far from the pad, preventing it from diving at wrong spots.
    # Reward = phi(next) - phi(prev):
    #   + when descending while centered (both factors improve)
    #   + when centering while low (x-improvement matters more near surface)
    #   - when rising or moving away from center
    prev_h = max(obs[1], 0.0)
    next_h = max(next_obs[1], 0.0)
    alpha = 5.0
    prev_phi = -prev_h * (2.718281828 ** (-alpha * obs[0]**2))
    next_phi = -next_h * (2.718281828 ** (-alpha * next_obs[0]**2))
    landing_approach = 2.0 * (next_phi - prev_phi)

    total_reward = goal_proximity + vel_penalty + upright_penalty + landing_approach

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "landing_approach": landing_approach
    }
    return total_reward, components
```

# 历史 iter_07 训练反馈
# Training Feedback

## Final-policy outcome
score=-5.371040, len=964.100000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-37.696217, 118.125687]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 95.903507 | 86.1% | 86.1% | 0.7% |
| upright_penalty | -5.454539 | -4.9% | 4.9% | 100.0% |
| goal_proximity | 4.753808 | 4.3% | 4.8% | 100.0% |
| velocity_penalty | -4.620901 | -4.1% | 4.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_07 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 7): landing_approach replaced with contact_bonus.
    The environment card identifies contact_success as a mandatory role —
    the only explicit "landing complete" signal. Iter 6's landing_approach
    was a continuous descent potential that never peaked at the landing event,
    providing only 0.974 ep_sum with zero successful landings (0/20).

    contact_bonus = 20 * centering_factor (moderate, between the dominating
    50 of iter 2 and the dead 8 of iter 4-5). No velocity gate — the
    velocity_penalty already enforces soft landing, and iter 3's velocity
    gate (exp(-3*speed²)) caused score collapse from 64→21 by making the
    bonus too hard to trigger.

    Velocity penalty kept at 0.3 (softened in iter 6 from 0.5) to avoid
    the "penalty wall" that prevented the agent from approaching the pad.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.3 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: moderate terminal reward for dual-leg landing ---
    # Restores the mandatory contact_success role (removed in iter 6).
    # Value 20 is chosen between the dominating 50 (iter 2) and dead 8 (iter 4-5).
    # centering_factor ensures off-pad landings receive proportionally less.
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 20.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return float(total_reward), components
```

# 修改自传（反思LLM过去每轮的修改决策和结果）
# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `?` → 分数 -76.6 → 63.7（+140.3）✅ 大幅改善
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 975→668

### v3：修改 `goal_proximity` → 分数 63.7 → 21.5（-42.3）❌ 明显退步
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 668→865

### v4：修改 `goal_proximity` → 分数 21.5 → -10.1（-31.6）❌ 明显退步
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 865→1000

### v5：修改 `goal_proximity` → 分数 -10.1 → -33.0（-22.9）❌ 明显退步
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 1000→1000

### v6：修改 `contact_bonus` → 分数 -33.0 → -81.8（-48.8）❌ 明显退步
- **你做了什么**：骨架重建：组件从 ['contact_bonus', 'goal_proximity', 'upright_penalty', 'velocity_penalty'] 变为 ['goal_proximity', 'landing_appro...
- **回合长度**：len: 1000→1000

### v7：修改 `contact_bonus` → 分数 -81.8 → -5.4（+76.4）✅ 大幅改善
- **你做了什么**：骨架重建：组件从 ['goal_proximity', 'landing_approach', 'upright_penalty', 'velocity_penalty'] 变为 ['contact_bonus', 'goal_proxim...
- **回合长度**：len: 1000→964

## 你的修改轨迹（按组件分组）

- **`goal_proximity`**：修改了 **3 次**，分数变化：v3(-42) → v4(-32) → v5(-23)
- **`contact_bonus`**：修改了 **2 次**，分数变化：v6(-49) → v7(+76)

**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + goal_proximity + upright_penalty + velocity_penalty | -76.56 | -76.56 | 0.00 | 975.15 | contact_bonus=13.313 goal_proximity=0.005 upright_penalty=-0.080 velocity_penalty=-0.134 | new_best |
| 2 | contact_bonus + goal_proximity + upright_penalty + velocity_penalty | 63.72 | 63.72 | 0.00 | 668.05 | contact_bonus=13.940 goal_proximity=0.009 upright_penalty=-0.084 velocity_penalty=-0.090 | new_best |
| 3 | contact_bonus + goal_proximity + upright_penalty + velocity_penalty | 21.45 | 63.72 | -42.27 | 864.55 | contact_bonus=13.007 goal_proximity=0.008 upright_penalty=-0.099 velocity_penalty=-0.085 | no_meaningful_improvement |
| 4 | contact_bonus + goal_proximity + upright_penalty + velocity_penalty | -10.11 | 63.72 | -73.83 | 1000.00 | contact_bonus=2.902 goal_proximity=0.008 upright_penalty=-0.069 velocity_penalty=-0.080 | no_meaningful_improvement |
| 5 | contact_bonus + goal_proximity + upright_penalty + velocity_penalty | -33.04 | 63.72 | -96.76 | 1000.00 | contact_bonus=3.032 goal_proximity=0.013 upright_penalty=-0.072 velocity_penalty=-0.067 | unsolved_high_achievement_continue_from_best |
| 6 | goal_proximity + landing_approach + upright_penalty + velocity_penalty | -81.80 | 63.72 | -145.52 | 1000.00 | goal_proximity=0.012 landing_approach=0.007 upright_penalty=-0.040 velocity_penalty=-0.031 | no_meaningful_improvement |
| 7 | contact_bonus + goal_proximity + upright_penalty + velocity_penalty | -5.37 | 63.72 | -69.09 | 964.10 | contact_bonus=5.793 goal_proximity=0.021 upright_penalty=-0.092 velocity_penalty=-0.073 | no_meaningful_improvement |


# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 2): velocity penalty is now height-gated via
    exp(-height / 2.0). At high altitude, velocity penalty is relaxed,
    allowing the agent to move freely toward the target. Near the ground,
    the penalty ramps up to full strength for soft landing.
    Goal proximity weight increased to 2.0 to strengthen approach incentive.
    """
    # --- goal proximity: encourage reducing Euclidean distance to pad ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 2.0 * (prev_dist - next_dist)   # delta reward (weight raised from 1.0)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)  # clamp to avoid amplification below pad
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: reward simultaneous leg contact, modulated by horizontal centering ---
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian-like decay with horizontal offset
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 50.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```
```
