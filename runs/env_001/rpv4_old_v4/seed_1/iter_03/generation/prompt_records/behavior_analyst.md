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
# 本轮训练反馈 (score=235.033692)
# Training Feedback

## Final-policy outcome
score=235.033692, len=425.200000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[82.814065, 278.428241]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 219.677227 | 87.8% | 87.8% | 71.7% |
| proximity_progress | 26.805481 | 10.7% | 11.0% | 96.7% |
| attitude_penalty | -2.012444 | -0.8% | 0.8% | 100.0% |
| landing_speed | -1.047543 | -0.4% | 0.4% | 68.9% |

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
| -115.0 | -123.9 | -176.9 | -34.9 | -0.5 | -25.9 | 18.1 | 232.7 | 250.0 | 245.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | -1.656 | -3.570 | -3.433 | -2.170 | -1.612 | -1.176 | -1.374 | -2.262 | -1.569 | -1.661 |
| landing_quality | 207.798 | 0.664 | 6.817 | 157.546 | 535.537 | 548.738 | 735.109 | 217.678 | 151.391 | 172.963 |
| landing_speed | -3.388 | -0.002 | -0.012 | -0.204 | -0.990 | -1.220 | -1.637 | -0.979 | -1.056 | -0.927 |
| proximity_progress | 8.946 | 7.699 | -0.453 | 18.356 | 25.213 | 22.297 | 26.852 | 26.355 | 26.588 | 26.827 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_quality | 77.9% | 2.8% | 3.9% | 52.8% | 80.0% | 82.9% | 85.8% | 71.9% | 66.2% | 67.7% |
| landing_speed | 77.9% | 2.8% | 3.9% | 52.8% | 80.0% | 82.9% | 85.7% | 69.1% | 62.4% | 64.3% |
| proximity_progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 96.7% | 95.6% | 96.0% |

# 本轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v2 — 语义修改：着陆反馈从接触门控改为高度门控。

    变更：
      - 删除 soft_landing_bonus（contact-gated 三重乘积，active_rate=0.3%）
      - 新增 landing_speed：近地速度惩罚，高度门控
      - 新增 landing_quality：近地品质奖励，鼓励低速+水平姿态
      - proximity_progress 系数 30→20，为着陆信号留权重空间
    """

    # ── 组件 A：接近进度（改善量信号） ──
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)  # 靠近>0, 远离<0

    # ── 组件 B：姿态惩罚（全程） ──
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ── 高度门控因子 ──
    # y_pos 从 0.5→0 时 gate 从 0→1 线性 ramp
    y_pos = next_obs[1]
    height_gate = max(0.0, min(1.0, (0.5 - y_pos) / 0.5))

    # ── 组件 C：近地速度惩罚 ──
    # 高度越低，速度惩罚越强；高空不约束速度
    speed_penalty = -0.5 * (next_obs[2] ** 2 + next_obs[3] ** 2)
    landing_speed = height_gate * speed_penalty

    # ── 组件 D：近地着陆品质奖励 ──
    # 鼓励低速 + 水平姿态，仅在近地区域生效
    speed_mag = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_quality = 2.718281828 ** (-4.0 * speed_mag)      # 速度越低越接近 1
    angle_quality = 2.718281828 ** (-4.0 * abs(angle))     # 角度越小越接近 1
    landing_quality = 1.5 * height_gate * speed_quality * angle_quality

    # ── 总奖励 ──
    total_reward = proximity_progress + attitude_penalty + landing_speed + landing_quality

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "landing_speed": landing_speed,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```

# 历史 iter_01 训练反馈
# Training Feedback

## Final-policy outcome
score=-2.976939, len=813.350000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-54.501032, 152.393336]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_progress | 36.742719 | 66.8% | 82.9% | 99.9% |
| soft_landing_bonus | 8.048444 | 14.6% | 14.6% | 0.3% |
| attitude_penalty | -1.348426 | -2.5% | 2.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_01 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v1 for the lunar‑lander‑like 2D vehicle landing task.
    - Physical components are built only from declared observation signals.
    - No terminal flags or hidden info are used.
    """
    # ---------- goal_proximity: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)   # positive when approaching

    # ---------- attitude_stabilization: penalty for tilt and angular velocity ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_contact: soft proxy for safe two-leg touch-down ----------
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    # Both legs must be in contact to be counted
    contact_score = left_contact * right_contact

    # Low‑velocity factor (both horizontal and vertical speed)
    vx = abs(next_obs[2])
    vy = abs(next_obs[3])
    velocity_factor = 1.0 / (1.0 + 10.0 * (vx + vy))

    # Upright orientation factor (angle near zero)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(angle))

    soft_landing_bonus = 5.0 * contact_score * velocity_factor * angle_factor

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```

# 历史 iter_02 训练反馈
# Training Feedback

## Final-policy outcome
score=235.033692, len=425.200000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[82.814065, 278.428241]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 219.677227 | 87.8% | 87.8% | 71.7% |
| proximity_progress | 26.805481 | 10.7% | 11.0% | 96.7% |
| attitude_penalty | -2.012444 | -0.8% | 0.8% | 100.0% |
| landing_speed | -1.047543 | -0.4% | 0.4% | 68.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_02 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v2 — 语义修改：着陆反馈从接触门控改为高度门控。

    变更：
      - 删除 soft_landing_bonus（contact-gated 三重乘积，active_rate=0.3%）
      - 新增 landing_speed：近地速度惩罚，高度门控
      - 新增 landing_quality：近地品质奖励，鼓励低速+水平姿态
      - proximity_progress 系数 30→20，为着陆信号留权重空间
    """

    # ── 组件 A：接近进度（改善量信号） ──
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)  # 靠近>0, 远离<0

    # ── 组件 B：姿态惩罚（全程） ──
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ── 高度门控因子 ──
    # y_pos 从 0.5→0 时 gate 从 0→1 线性 ramp
    y_pos = next_obs[1]
    height_gate = max(0.0, min(1.0, (0.5 - y_pos) / 0.5))

    # ── 组件 C：近地速度惩罚 ──
    # 高度越低，速度惩罚越强；高空不约束速度
    speed_penalty = -0.5 * (next_obs[2] ** 2 + next_obs[3] ** 2)
    landing_speed = height_gate * speed_penalty

    # ── 组件 D：近地着陆品质奖励 ──
    # 鼓励低速 + 水平姿态，仅在近地区域生效
    speed_mag = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_quality = 2.718281828 ** (-4.0 * speed_mag)      # 速度越低越接近 1
    angle_quality = 2.718281828 ** (-4.0 * abs(angle))     # 角度越小越接近 1
    landing_quality = 1.5 * height_gate * speed_quality * angle_quality

    # ── 总奖励 ──
    total_reward = proximity_progress + attitude_penalty + landing_speed + landing_quality

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "landing_speed": landing_speed,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```

# 修改自传（反思LLM过去每轮的修改决策和结果）
# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `?` → 分数 -3.0 → 235.0（+238.0）✅ 大幅改善
- **你做了什么**：骨架重建：组件从 ['attitude_penalty', 'proximity_progress', 'soft_landing_bonus'] 变为 ['attitude_penalty', 'landing_quality', 'la...
- **回合长度**：len: 813→425


**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | attitude_penalty + proximity_progress + soft_landing_bonus | -2.98 | -2.98 | 0.00 | 813.35 | attitude_penalty=-0.017 proximity_progress=0.257 soft_landing_bonus=1.013 | new_best |
| 2 | attitude_penalty + landing_quality + landing_speed + proximity_progress | 235.03 | 235.03 | 0.00 | 425.20 | attitude_penalty=-0.016 landing_quality=0.764 landing_speed=-0.016 proximity_progress=0.056 | target_solved_new_best |

```
