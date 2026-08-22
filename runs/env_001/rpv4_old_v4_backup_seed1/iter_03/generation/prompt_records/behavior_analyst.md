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
# 本轮训练反馈 (score=63.720709)
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
| -73.8 | -168.1 | -119.8 | -97.7 | -92.9 | -2.9 | 70.8 | -44.1 | -17.7 | 48.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 46.256 | 0.000 | 2.878 | 16.155 | 67.625 | 1049.699 | 4038.918 | 1514.911 | 675.163 | 746.690 |
| goal_proximity | 2.241 | 0.328 | 1.020 | 1.168 | 1.391 | 1.615 | 1.992 | 1.421 | 1.478 | 1.856 |
| upright_penalty | -10.327 | -2.963 | -3.341 | -3.688 | -7.615 | -8.324 | -8.611 | -7.933 | -8.543 | -7.125 |
| velocity_penalty | -32.668 | -9.988 | -10.782 | -9.769 | -9.524 | -9.912 | -8.834 | -10.570 | -11.150 | -8.938 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 2.0% | 0.0% | 0.3% | 0.2% | 1.3% | 3.4% | 16.0% | 5.3% | 4.0% | 3.7% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 99.6% | 99.5% | 99.4% | 99.1% |
| upright_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 99.6% | 99.6% | 99.4% | 99.1% |

# 本轮奖励函数代码
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

# 修改自传（反思LLM过去每轮的修改决策和结果）
# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `?` → 分数 -76.6 → 63.7（+140.3）✅ 大幅改善
- **你做了什么**：(响应格式异常，无法解析修改内容)
- **回合长度**：len: 975→668


**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + goal_proximity + upright_penalty + velocity_penalty | -76.56 | -76.56 | 0.00 | 975.15 | contact_bonus=13.313 goal_proximity=0.005 upright_penalty=-0.080 velocity_penalty=-0.134 | new_best |
| 2 | contact_bonus + goal_proximity + upright_penalty + velocity_penalty | 63.72 | 63.72 | 0.00 | 668.05 | contact_bonus=13.940 goal_proximity=0.009 upright_penalty=-0.084 velocity_penalty=-0.090 | new_best |

```
