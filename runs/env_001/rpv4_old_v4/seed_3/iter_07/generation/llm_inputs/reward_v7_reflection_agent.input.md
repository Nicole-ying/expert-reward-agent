第 6 轮迭代训练已经结束，得分 -15.733251（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(6)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

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

实验记录位于 `rpv4_old_v4/seed_3/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件

现在开始调查。

# 上一轮训练反馈 (iter_6, score=-15.733251)
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
| -61.3 | -269.8 | -103.8 | -9.2 | -15.6 | -20.2 | -6.5 | 156.2 | 89.4 | -15.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1168.992 | 136.610 | 0.000 |
| descent_incentive | 1.494 | 0.065 | 0.835 | 3.442 | 2.908 | 2.386 | 3.369 | 3.680 | 3.940 | 3.329 |
| goal_proximity | 5.713 | -10.904 | 4.405 | 7.912 | 7.854 | 7.817 | 7.933 | 7.953 | 7.948 | 7.852 |
| orientation_penalty | -4.615 | -2.576 | -3.730 | -3.760 | -3.824 | -4.027 | -4.241 | -2.803 | -3.661 | -3.858 |
| velocity_penalty | -3.021 | -3.908 | -4.338 | -4.669 | -5.162 | -5.656 | -6.045 | -5.631 | -6.602 | -5.670 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 37.3% | 4.6% | 0.0% |
| descent_incentive | 66.9% | 20.0% | 37.9% | 79.0% | 78.0% | 78.2% | 78.1% | 69.1% | 75.3% | 73.9% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.1% | 99.2% | 100.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.3% | 99.3% | 100.0% |

# 上一轮奖励函数代码
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

# Subagent 数据报告
# 训练数据报告

## 1. 行为事实

- 得分：score=-15.73，range=[-41.62, 19.63]
- 终止方式：terminated=0/20（成功落地 0 次），truncated=20/20（全部达到 1000 步时间上限）
- 回合长度：len=1000.00（全部截断）
- reward_errors：0
- Checkpoint 轨迹：cp10%=-61.3 → cp40%=-9.2 → cp80%=156.2（峰值）→ cp90%=89.4 → cp100%=-15.0（最终得分 100 步窗口内剧烈回撤）
- contact_bonus checkpoint 轨迹：cp10%-cp70% 均为 0.000，cp80%=1168.992（骤升），cp90%=136.610（骤降），cp100%=0.000（归零）
- descent_incentive 本轮 ep_sum_mean=3.366，active_rate=73.8%（未全覆盖）
- 早期终止（<150 步且 score<-50）：0/20

## 2. 骨架演进表

| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | orientation_penalty_ep_sum | orientation_penalty_active | descent_incentive_ep_sum | descent_incentive_active | contact_bonus_ep_sum | contact_bonus_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -32.35 | 1000.00 | 0/20/20 | 3.758 | 100.0% | -0.486 | 100.0% | -3.959 | 100.0% | — | — | 0.000 | 0.0% |
| 2 | 107.13 | 775.60 | 11/9/20 | 3.963 | 99.1% | -5.313 | 99.2% | -3.090 | 100.0% | — | — | 154.465 | 2.8% |
| 3 | -26.17 | 1000.00 | 0/20/20 | 7.685 | 100.0% | -4.758 | 100.0% | -4.038 | 100.0% | — | — | 0.000 | 0.0% |
| 4 | 183.52 | 710.25 | 19/1/20 | 7.951 | 98.1% | -5.736 | 98.4% | -3.386 | 100.0% | 0.779 | 72.4% | 114.083 | 4.9% |
| 5 | -15.01 | 1000.00 | 0/20/20 | 7.881 | 100.0% | -6.826 | 100.0% | -3.615 | 100.0% | 0.624 | 66.1% | 0.000 | 0.0% |
| 6 | -15.73 | 1000.00 | 0/20/20 | 7.861 | 100.0% | -5.725 | 100.0% | -3.917 | 100.0% | 3.366 | 73.8% | 0.000 | 0.0% |

## 3. 数字标记

**死组件**：
- contact_bonus：active_rate=0.0%，ep_sum_mean=0.000（连续 3 轮在非成功运行中均为 0%）

**量级对比**（ep_sum ÷ len 每步均值）：
- 最大正分量：goal_proximity=0.00786，descent_incentive=0.00337
- 最大负分量：velocity_penalty=-0.00572，orientation_penalty=-0.00392
- 最小分量：contact_bonus=0.00000

**修改变化（Δ iter_05→iter_06）**：
| component | Δ ep_sum_mean | Δ active_rate |
|---|---:|---:|
| goal_proximity | -0.021 | 0.0pp |
| velocity_penalty | +1.102 | 0.0pp |
| orientation_penalty | -0.302 | 0.0pp |
| descent_incentive | +2.742 | +7.7pp |
| contact_bonus | 0.000 | 0.0pp |

**无主导**：goal_proximity magnitude_share=39.1%，未超过 80%。
**无淹没**：所有组件 active_rate 均在 0%-100% 之间，未触发淹没条件。
**无全覆盖惩罚**：无组件同时满足 active_rate>95% 且 signed_share 为负。

## 4. 语义演化（跨轮组件功能追踪）

### 接近/进度信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近信号 | v1_goal_proximity → v2_goal_proximity → v3_goal_proximity → v4_goal_proximity → v5_goal_proximity → v6_goal_proximity | -32.3 → 107.1 → -26.2 → 183.5 → -15.0 → -15.7 | 改善量：0次切换（始终为平方距离delta） |

goal_proximity 在所有 6 轮中保持"改善量 delta"形态未变。ep_sum 在迭代 3 因权重翻倍（w_x 1→2, w_y 2→4）从 ~3.8 跳升至 ~7.7，此后稳定在 7.8-8.0 区间，是每轮最稳定的组件。

### 稳定性/约束信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 速度约束 | v1_velocity_penalty → v2_velocity_penalty → v3+v4+v5+v6_velocity_penalty | -32.3 → 107.1 → -26.2 → 183.5 → -15.0 → -15.7 | 门控形式：1次切换（v1曼哈顿距离门控→v2+指数高度门控） |
| 姿态约束 | v1_orientation_penalty → v2+v3+v4+v5+v6_orientation_penalty | -32.3 → 107.1 → -26.2 → 183.5 → -15.0 → -15.7 | 0次切换（始终为|angle|+angvel²） |

velocity_penalty 在 v1→v2 完成从曼哈顿距离门控到指数高度门控的 1 次切换后，后续 5 轮未再变更设计。ep_sum 在成功轮次（v2=-5.31, v4=-5.74）和失败轮次（v5=-6.83, v6=-5.72）之间波动，失败轮次中惩罚更强。orientation_penalty 6 轮 0 次设计变更。

### 着陆/接触信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆奖励 | v1_contact_bonus → v2_contact_bonus → v3_contact_bonus → v4_contact_bonus → v5_contact_bonus → v6_contact_bonus | -32.3 → 107.1 → -26.2 → 183.5 → -15.0 → -15.7 | 稀疏bonus：权重反复：w=10→10→4→4→5→4，共3次切换 |
| 下降引导 | — → — → — → v4_descent_incentive → v5_descent_incentive → v6_descent_incentive | — → — → — → 183.5 → -15.0 → -15.7 | 门控→全高度：1次切换（v4/v5 y<0.8→v6 y<1.5） |

contact_bonus 在 6 轮中经历 3 次权重切换（10→10→4→4→5→4），仅在 iter_02 和 iter_04 两轮成功激活（active_rate 2.8% 和 4.9%），其余 4 轮均为死组件（active_rate 0%）。checkpoint 数据显示 iter_06 中 cp80% 曾达到 1168.992（骤升），但在 cp90% 暴跌至 136.610，cp100% 归零——agent 在训练中期发现着陆后最终放弃了该行为。

descent_incentive 从 v4 引入至今经历 1 次设计切换（门控高度从 0.8 扩大到 1.5，权重从 2.0 提升到 6.0）。v6 中 ep_sum 从 v5 的 0.624 增长到 3.366（+438%），active_rate 从 66.1% 提升到 73.8%，但并未转化为着陆成功（terminated 保持 0/20）。

### 完成/成功代理
无独立完成代理组件。contact_bonus 兼作稀疏成功信号。

# 历史最佳奖励函数代码
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

# 环境事实（只据此理解任务和变量）
## 1. 任务目标
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: 推测为 float32  
- obs[0]: x_position，水平坐标（相对于目标平台的 x 偏移），reward_usable: true  
- obs[1]: y_position，垂直坐标（相对于平台高度，即飞行器底部到平台表面的距离），reward_usable: true  
- obs[2]: x_velocity，水平线速度，reward_usable: true  
- obs[3]: y_velocity，垂直线速度，reward_usable: true  
- obs[4]: body_angle，身体倾斜角度（弧度 or 归一化角度），reward_usable: true  
- obs[5]: angular_velocity，角速度，reward_usable: true  
- obs[6]: left_support_contact，左支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true  
- obs[7]: right_support_contact，右支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true  

备注：所有维度均为连续数字，接触标志虽为 0/1 但以浮点形式给出。

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine（不点火，靠惯性/重力运动）  
- action 1: left_orientation_engine（点火左向姿态调节引擎，产生向左旋转力矩）  
- action 2: main_engine（点火主引擎，产生向上的推力，通常用于减速/悬停）  
- action 3: right_orientation_engine（点火右向姿态调节引擎，产生向右旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 可能代表身体已静止，结合 y_position≈0 且左右支撑接触为 1 时可视为成功着陆。但 termination 本身是模糊的，因为 crash 后也可能导致静止。  
- failure-like termination: crash_or_body_contact 中“crash”部分，horizontal_position_outside_viewport（水平超出视口）  
- ambiguous termination: body_not_awake_or_settled 的具体触发条件未知，可能是身体进入休眠/静止状态；crash_or_body_contact 中的 body_contact 可能包含正常着陆接触，需依赖位置判断。  
- truncation: 未提供  

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无 success 字段）  
- explicit_failure_flag_available: false（同样无 failure 字段）  
- allowed_info_fields: 无（info 返回空字典 {}）  
- forbidden_or_uncertain_info_fields: 所有 internal 字段均不可用，如 termination_reason 等

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4]), angular_velocity (obs[5])
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: action 值（0~3），可用于燃料惩罚或推力鼓励
- other: 通过 next_obs 计算上述量的变化率（如速度变化、位置变化）