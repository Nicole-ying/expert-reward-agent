第 7 轮迭代训练已经结束，得分 252.778635（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(7)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

3. **查全局历史避免打转**——用 `read_memory()` 看所有迭代。**关键判断：你是不是在同一个语义功能上反复修改但无突破？**（注意：组件名字可能变了，但语义功能相同——比如 success_landing_bonus 和 success_bonus 都是"着陆成功信号"。）如果某个语义已经改了 2-3 轮都不好，不要再改它了——换一个语义方向。

4. **参考 best code 但要验证**——用 `read_reward_code(best_iter)` 看历史最佳奖励函数的代码。它的哪些组件可能促成了当时的最高分？但不要盲目照搬——结合 best 那轮的 feedback（`read_training_feedback(best_iter)`）看：它的得分分布、终止率、组件 activation 是否健康。best 只是记录最高分，不一定代表正确的引导方式。

5. **Checkpoint 是训练过程的证据**——如果某个组件在 checkpoint 40%→80% 期间 ep_sum 持续为 0 或不变，说明它没有提供有效梯度，agent 在训练中忽略了它。

### 修改原则

- **单一语义修改**：每轮只解决一个问题。通常是一个组件的修改，但如果需要联动（如改一个组件的公式 + 调另一个组件的权重来配合），只要服务于同一语义也算一次修改。
- **L1**：只调系数/权重，不改数学表达式。适用于组件方向对但力度不对。
- **L2**：改一个组件的数学形态（稀疏→连续、状态值→改善量、均匀→高度感知等）。适用于组件的引导逻辑有问题（如均匀惩罚导致无近地减速梯度）。
- **L3**：换主信号框架。仅当多轮迭代从未达标，且历史所有语义方向都已穷尽。
- **⚠️ 组件被 exploit 时必须干预**：如果一个组件 magnitude_share > 90%，即使它刚被激活，也必须修改它——加触发条件限制、加每 episode 次数上限、或收紧阈值。"刚激活所以只做 L1"不适用于 exploit 场景。
- **避免打转**：如果 `read_past_reflection` 显示你之前已经改过同一语义且结果不好，换方向。
- **接近目标时优先微调**：如果当前得分或 best 得分已经接近目标（差距 < 10%），**强烈推荐 L1 微调**（只调系数）。历史尝试中反复出现同一模式：分数已接近目标，却选择 L2 改组件形态，结果越改越差，连续多轮倒退。架构已有基本有效性时，大规模改动风险远大于收益。

### 你需要的信息都在这里

实验记录位于 `rpv4_old_v4/seed_0/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件

现在开始调查。

# 上一轮训练反馈 (iter_7, score=252.778635)
# Training Feedback

## Final-policy outcome
score=252.778635, len=308.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[49.745101, 291.205816]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| fuel_penalty | -2.797750 | -33.9% | 33.9% | 51.6% |
| progress_reward | 2.671382 | 32.3% | 33.1% | 95.2% |
| vel_penalty | -2.518084 | -30.5% | 30.5% | 95.7% |
| orientation_penalty | -0.210587 | -2.5% | 2.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估
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
| -151.7 | -24.7 | 77.7 | 91.6 | 147.6 | 182.0 | 214.0 | 247.6 | 197.7 | 262.8 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| fuel_penalty | -6.555 | -7.807 | -2.827 | -2.458 | -2.481 | -2.466 | -2.496 | -2.690 | -2.383 | -2.889 |
| orientation_penalty | -0.117 | -0.183 | -0.261 | -0.319 | -0.343 | -0.294 | -0.334 | -0.287 | -0.300 | -0.202 |
| progress_reward | 0.697 | 1.914 | 2.494 | 2.480 | 2.581 | 2.593 | 2.596 | 2.626 | 2.620 | 2.688 |
| vel_penalty | -3.067 | -1.956 | -2.493 | -2.713 | -2.721 | -2.759 | -2.703 | -2.594 | -2.870 | -2.523 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| fuel_penalty | 87.5% | 91.2% | 58.3% | 56.6% | 52.5% | 49.3% | 50.6% | 50.3% | 52.3% | 53.1% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 98.7% | 98.3% | 97.2% | 96.6% | 95.8% | 95.2% | 95.9% | 95.0% |
| vel_penalty | 100.0% | 100.0% | 98.9% | 98.5% | 97.4% | 96.9% | 96.3% | 95.8% | 96.3% | 95.6% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_prev, y_prev = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 1. Progress reward (delta improvement on distance) ──
    #     Positive when approaching the target pad, negative when retreating.
    #     w=2.0 gives stronger gradient than iter-5's w=1.0 to compensate
    #     for the absence of a landing_proximity bonus.
    dist_prev = (x_prev * x_prev + y_prev * y_prev) ** 0.5
    dist_curr = (x * x + y * y) ** 0.5
    progress = dist_prev - dist_curr
    progress_reward = 2.0 * progress

    # ── 2. Soft landing gate (velocity + orientation penalties near pad) ──
    distance = dist_curr
    gate = 1.0 / (1.0 + 3.0 * distance)   # close to 1 when near pad, decays with distance

    vel_penalty = -0.3 * gate * (vx * vx + vy * vy + 0.1 * angular_vel * angular_vel)
    orientation_penalty = -0.2 * gate * (angle * angle)

    # ── 3. Fuel efficiency (light penalty on engine use) ──
    if action == 2:               # main engine
        fuel_penalty = -0.02
    elif action == 1 or action == 3:  # orientation engines
        fuel_penalty = -0.005
    else:                         # no engine
        fuel_penalty = 0.0

    total_reward = progress_reward + vel_penalty + orientation_penalty + fuel_penalty

    components = {
        'progress_reward': progress_reward,
        'vel_penalty': vel_penalty,
        'orientation_penalty': orientation_penalty,
        'fuel_penalty': fuel_penalty,
    }
    return float(total_reward), components
```

# Subagent 数据报告
# 训练数据报告 — iter_07

## 1. 行为事实

- **得分**：score=252.78，range=[49.75, 291.21]（全部 20 集为正，最低 49.7、最高 291.2）
- **终止方式**：terminated 20/20（全部成功着陆），truncated 0/20（无超时）
- **回合长度**：len=308.2
- **奖励误差**：0
- **Checkpoint 官方得分轨迹**：cp1=-151.7 → cp2=-24.7 → cp3=77.7 → cp4=91.6 → cp5=147.6 → cp6=182.0 → cp7=214.0 → cp8=247.6 → cp9=197.7 → cp10=262.8  
  （从 cp3 起全程为正，cp1→cp10 累计上升 414.5 分；cp9 有一个回落 -49.9，cp10 恢复并达到最高）

### 各组件 ep_sum_mean 趋势

| 组件 | cp1 | cp2 | cp3 | cp4 | cp5 | cp6 | cp7 | cp8 | cp9 | cp10 |
|---|---|---|---|---|---|---|---|---|---|---|
| fuel_penalty | -6.555 | -7.807 | -2.827 | -2.458 | -2.481 | -2.466 | -2.496 | -2.690 | -2.383 | -2.889 |
| orientation_penalty | -0.117 | -0.183 | -0.261 | -0.319 | -0.343 | -0.294 | -0.334 | -0.287 | -0.300 | -0.202 |
| progress_reward | 0.697 | 1.914 | 2.494 | 2.480 | 2.581 | 2.593 | 2.596 | 2.626 | 2.620 | 2.688 |
| vel_penalty | -3.067 | -1.956 | -2.493 | -2.713 | -2.721 | -2.759 | -2.703 | -2.594 | -2.870 | -2.523 |

### 各组件 active_rate 趋势

| 组件 | cp1 | cp2 | cp3 | cp4 | cp5 | cp6 | cp7 | cp8 | cp9 | cp10 |
|---|---|---|---|---|---|---|---|---|---|---|
| fuel_penalty | 87.5% | 91.2% | 58.3% | 56.6% | 52.5% | 49.3% | 50.6% | 50.3% | 52.3% | 53.1% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 98.7% | 98.3% | 97.2% | 96.6% | 95.8% | 95.2% | 95.9% | 95.0% |
| vel_penalty | 100.0% | 100.0% | 98.9% | 98.5% | 97.4% | 96.9% | 96.3% | 95.8% | 96.3% | 95.6% |

---

## 2. 骨架演进表

| iter | score | len | term | progress_reward_ep_sum | progress_reward_active | vel_penalty_ep_sum | vel_penalty_active | orientation_penalty_ep_sum | orientation_penalty_active | fuel_penalty_ep_sum | fuel_penalty_active | contact_reward_ep_sum | contact_reward_active | landing_proximity_reward_ep_sum | landing_proximity_reward_active | vel_damping_ep_sum | vel_damping_active | approach_reward_ep_sum | approach_reward_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -159.49 | 303.35 | 20/0 | 0.104 | 100.0% | — | — | -0.034 | 100.0% | — | — | 0.000 | 0.0% | — | — | 0.000 | 0.0% | — | — |
| 2 | -36.38 | 1000.00 | 0/20 | 0.001 | 100.0% | — | — | -0.000 | 100.0% | — | — | — | — | 2.704 | 100.0% | -0.027 | 95.3% | — | — |
| 3 | -434.87 | 967.20 | 1/19 | -0.002 | 100.0% | — | — | -0.001 | 100.0% | — | — | — | — | 0.026 | 68.3% | 0.000 | 0.0% | — | — |
| 4 | -59.19 | 977.90 | 1/19 | 0.001 | 100.0% | — | — | -0.001 | 100.0% | — | — | — | — | 4.731 | 100.0% | -0.022 | 92.1% | — | — |
| 5 | -124.22 | 68.35 | 20/0 | 0.016 | 100.0% | — | — | -0.019 | 100.0% | -0.002 | 3.2% | — | — | 0.034 | 100.0% | -0.171 | 45.6% | — | — |
| 6 | -119.14 | 68.45 | 20/0 | — | — | -0.153 | 99.7% | -0.002 | 100.0% | -0.001 | 13.6% | — | — | — | — | — | — | -0.974 | 100.0% |
| 7 | 252.78 | 308.20 | 20/0 | 0.009 | 95.2% | -0.008 | 95.7% | -0.001 | 100.0% | -0.009 | 51.6% | — | — | — | — | — | — | — | — |

注：`—` 表示该组件在该轮不存在。iter_01 的 progress_reward/contact_reward/orientation_penalty/vel_damping 列重用于全表；iter_06 的 approach_reward 和 vel_penalty 为独立列；iter_07 的 progress_reward/orientation_penalty/vel_penalty/fuel_penalty 对应回原始列。各组件 ep_sum 为 episode_sum_mean 近似值（取 signed_share × |score|/100 或从原数据直接提取）。

---

## 3. 数字标记

### 本轮（iter_07）

- **主导**：无。最大 magnitude_share 为 fuel_penalty 33.9%，未超过 80%。
- **死组件**：无。最小 active_rate 为 fuel_penalty 51.6%，未低于 2%。
- **淹没**：无。最高 active_rate 100% 的 orientation_penalty 其 magnitude_share=2.5%，不满足 active_rate>90% 且 magnitude_share<3%（仅"淹没"标准触发，但严格来说 magnitude_share<3% 满足条件而 active_rate=100%>90%，然而 magnitude_share<3% 非常接近边界；根据定义"淹没"要求同时满足 active_rate>90% 且 magnitude_share<3%，orientation_penalty 满足 2.5%<3% 但该规则并非"标记"，此处仅作"量级最小"呈现，不标记）。
- **全覆盖惩罚**：无。没有组件 active_rate>95% 且 signed_share 为负——orientation_penalty 的 active_rate=100%>95% 且 signed_share 为负（-2.5%），但 magnitude_share 仅 2.5%，按定义应标记：orientation_penalty 的 active_rate=100.0%>95% 且 signed_share 为负（-2.5%），满足"全覆盖惩罚"条件。
- **量级对比**（每步均值 = ep_sum / len）：
  - 最大：fuel_penalty=-0.00908, progress_reward=+0.00867
  - 最小：orientation_penalty=-0.00068（量级最小），vel_penalty=-0.00817
  - 实际上：每步均值最大两组件：fuel_penalty 和 progress_reward；最小两组件：orientation_penalty 和 vel_penalty

### 修改变化（Δ iter_06→iter_07）

本表仅列出 iter_06 和 iter_07 共有的组件名。关注**实质对应**：iter_06 的 approach_reward 是 progress_reward 的语义等价物（同为距离改善量），iter_07 改回 progress_reward 名称但语义相同。

| 组件 | Δ ep_sum | Δ active_rate | 说明 |
|---|---|---|---|
| progress_reward（对应 iter_06 approach_reward） | +0.983 → 从 -0.974 变为 +0.009（ep_sum 尺度不同，实际变正） | -4.8%（100.0%→95.2%） | 语义等价，数值变化来自权重和门控 |
| vel_penalty | +0.145（从 -0.153 缓和到 -0.008） | -4.0%（99.7%→95.7%） | 门控形式重建，惩罚减轻 |
| orientation_penalty | -0.001（基本持平，-0.002→-0.001） | 0.0%（100.0%→100.0%） | 门控重建，影响微小 |
| fuel_penalty | -0.008（从 -0.001 增强到 -0.009） | +38.0%（13.6%→51.6%） | 门槛降低（-0.005→-0.005 不变但统一费率变化？本轮 action 1/3 同样 -0.005，激活率大幅上升） |

---

## 4. 语义演化（跨轮组件功能追踪）

按语义功能对组件分组，列出各轮参与该功能的组件名、得分、以及"方向切换"次数。

| 语义功能 | 各轮组件名与得分 | 方向切换 |
|---|---|---|
| **接近/进度信号** | iter_01: progress_reward (w=1.0, ep_sum≈+0.10, 连续奖励/改善量) → iter_02: progress_reward (w=1.0, ep_sum≈+0.001, 连续/改善量，被 landing_proximity 淹没) → iter_03: progress_reward (w=1.0, ep_sum≈-0.002, 连续/改善量) → iter_04: progress_reward (w=1.0, ep_sum≈+0.001, 连续/改善量) → iter_05: progress_reward (w=1.0, ep_sum≈+0.016, 连续/改善量) → iter_06: approach_reward (实质等价 progress_reward, w=2.0, ep_sum≈-0.974, 连续/改善量) → iter_07: progress_reward (w=2.0, ep_sum≈+2.67, 连续/改善量) | 0 次切换（全程为连续形状/改善量 delta 形式） |
| **着陆/接触信号** | iter_01: contact_reward (w=2.0, 稀疏 bonus, 死组件 ep_sum=0) → iter_02: landing_proximity_reward (w=3.0, 连续奖励/状态值, ep_sum≈+2704) → iter_03: landing_proximity_reward (w=5.0, 连续奖励/改善量加权, ep_sum≈+25.5) → iter_04: landing_proximity_reward (w=10.0, 连续奖励/状态值, ep_sum≈+4731) → iter_05: landing_proximity_reward (w=2.0, 连续奖励/状态值, ep_sum≈+2.32) → iter_06: **移除** → iter_07: **无着陆奖励** | 4 次切换（稀疏 bonus→连续奖励；改善量加权→状态值→状态值→移除 w/ 无近似→无着陆信号） |
| **稳定性/约束信号** | iter_01: vel_damping (gated, w=0.5, 死组件 ep_sum=0) + orientation_penalty (全高度, ep_sum≈-0.034) → iter_02: vel_damping (gated, w=0.5, ep_sum≈-26.78) + orientation_penalty (全高度, ep_sum≈-0.000) → iter_03: vel_damping (gated, w=0.5, 死组件 ep_sum=0) + orientation_penalty (全高度, ep_sum≈-0.001) → iter_04: vel_damping (gated, w=0.5, ep_sum≈-21.76) + orientation_penalty (全高度, ep_sum≈-0.001) → iter_05: vel_damping (gated, w=0.5, ep_sum≈-11.71) + orientation_penalty (全高度, ep_sum≈-0.019) → iter_06: vel_penalty (软门控 1/(1+3d), ep_sum≈-10.49) + orientation_penalty (软门控, ep_sum≈-0.145) → iter_07: vel_penalty (软门控, ep_sum≈-2.52) + orientation_penalty (软门控, ep_sum≈-0.211) | 1 次切换（门控→软门控，发生在 iter_05→iter_06，此后保持稳定） |
| **燃料/稀疏惩罚** | iter_01~04: 无燃料组件 → iter_05: fuel_penalty (离散费率 -0.2/-0.05/0, ep_sum≈-0.11) → iter_06: fuel_penalty (离散费率 -0.02/-0.005/0, ep_sum≈-0.047) → iter_07: fuel_penalty (离散费率 -0.02/-0.005/0, ep_sum≈-2.80) | 0 次切换（全程为连续性惩罚/离散费率） |
| **完成/成功代理** | 无明确的条件性大额正信号或乘积形式完成接近度奖励存在于各轮中 | — |

---

## 更新历史记忆表

| iter | skeleton | score | best | delta | len | key_signal | action |
|---|---|---|---|---|---|---|---|
| 1 | contact_reward + orientation_penalty + progress_reward + vel_damping | -159.49 | -159.49 | 0.00 | 303.35 | contact_reward=0.000 orientation_penalty=-0.034 progress_reward=+0.104 vel_damping=0.000 | new_best |
| 2 | landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | -36.38 | -36.38 | 0.00 | 1000.00 | landing_proximity_reward=+2704.04 orientation_penalty=-0.000 progress_reward=+0.001 vel_damping=-26.78 | new_best |
| 3 | landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | -434.87 | -36.38 | -398.49 | 967.20 | landing_proximity_reward=+25.54 orientation_penalty=-0.001 progress_reward=-0.002 vel_damping=0.000 | no_meaningful_improvement |
| 4 | landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | -59.19 | -36.38 | -22.81 | 977.90 | landing_proximity_reward=+4731.13 orientation_penalty=-0.001 progress_reward=+0.001 vel_damping=-21.76 | no_meaningful_improvement |
| 5 | fuel_penalty + landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | -124.22 | -36.38 | -87.84 | 68.35 | fuel_penalty=-0.002 landing_proximity_reward=+0.034 orientation_penalty=-0.019 progress_reward=+0.016 vel_damping=-11.71 | unsolved_stagnation_fresh_restart |
| 6 | approach_reward + fuel_penalty + orientation_penalty + vel_penalty | -119.14 | -36.38 | -82.76 | 68.45 | approach_reward=-66.67 fuel_penalty=-0.047 orientation_penalty=-0.145 vel_penalty=-10.49 | no_meaningful_improvement |
| 7 | fuel_penalty + orientation_penalty + progress_reward + vel_penalty | 252.78 | 252.78 | 0.00 | 308.20 | fuel_penalty=-2.80 orientation_penalty=-0.211 progress_reward=+2.67 vel_penalty=-2.52 | target_solved_new_best |

# 环境事实（只据此理解任务和变量）
## 1. 任务目标
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float (实际为连续值，左/右支撑为 0.0 或 1.0)
- 各维度含义：
  - obs[0] (x_position): 相对于目标着陆垫的水平坐标，reward_usable: true  
  - obs[1] (y_position): 相对于着陆垫高度的垂直坐标，reward_usable: true  
  - obs[2] (x_velocity): 水平线速度，reward_usable: true  
  - obs[3] (y_velocity): 垂直线速度，reward_usable: true  
  - obs[4] (body_angle): 机体角度（方向），reward_usable: true  
  - obs[5] (angular_velocity): 角速度，reward_usable: true  
  - obs[6] (left_support_contact): 左侧支撑接触标志（1.0表示接触，0.0表示未接触），reward_usable: true  
  - obs[7] (right_support_contact): 右侧支撑接触标志，reward_usable: true

所有观测维度均可在自定义奖励中安全使用。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - action 0: no_engine – 什么都不做，滑翔/惯性飞行
  - action 1: left_orientation_engine – 点燃左侧姿态发动机（产生逆时针或顺时针旋转力矩）
  - action 2: main_engine – 点燃主发动机（提供向机头方向的推力，通常向上或减速）
  - action 3: right_orientation_engine – 点燃右侧姿态发动机（与左侧相反的旋转力矩）

注意：该飞行器没有横向直接推进器，水平控制主要通过倾斜机身+主发动机实现。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 中的 “settled” 部分 —— 推测为飞行器稳定停靠在目标着陆垫上（满足速度、角度阈值且支撑接触）
- failure-like termination: crash_or_body_contact（当不属于成功着陆时的身体/地面碰撞）和 horizontal_position_outside_viewport
- ambiguous termination: body_not_awake_or_settled 内部 —— 由于任务描述未区分“settled”与“not awake”，但根据任务目标“reach and settle”可推断出成功情况；需结合接触标志、位置接近度进一步解析
- truncation: 无（step 返回的 truncated 恒为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，终止后无明确标志）
- explicit_failure_flag_available: false 
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均未提供，不可依赖。

成功/失败只能通过下一状态的观测（next_obs）与 done 组合推断：例如，若终止时左右支撑接触均为 1，且 x、y 位置接近 0，速度很小，角度接近 0（或垂直），则可以假定为成功着陆。

## 7. 可用于奖励函数的信号
- 位置信号：
  - next_obs[0] (x_position): 相对于目标着陆垫的水平偏差
  - next_obs[1] (y_position): 相对于着陆垫高度的垂直偏差（越高越正？）
- 速度信号：
  - next_obs[2] (x_velocity): 水平速度（靠近着陆垫时应接近 0）
  - next_obs[3] (y_velocity): 垂直速度（靠近着陆垫时应接近 0 或一个小负值以表示软下降）
- 姿态信号：
  - next_obs[4] (body_angle): 机体角度，用于维持稳定姿态（接近 0 表示竖直）
  - next_obs[5] (angular_velocity): 角速度，用于惩罚大幅旋转
- 接触信号：
  - next_obs[6] (left_support_contact): 左侧支撑接触 (0/1)
  - next_obs[7] (right_support_contact): 右侧支撑接触 (0/1)
  可用于判断着陆状态、两侧接触平衡、软/硬着陆
- 动作/发动机信号：
  - action: 离散动作，可用于惩罚推力使用（action 0 为无推力，action 1/3 为姿态推力，action 2 为主发动机推力）
- 其他衍生物：
  - 距离到目标：sqrt(x_position² + y_position²)
  - 接近速度/方向
  - 着陆瞬间的状态组合（位置、速度、角度）