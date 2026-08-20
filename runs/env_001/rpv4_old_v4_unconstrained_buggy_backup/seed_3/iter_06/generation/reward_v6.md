现在我已经全面了解了历史。让我进行诊断。

---

## 诊断报告

### Evidence

| 维度 | iter_2 (best=72.90) | iter_5 (当前=-96.02) | 解读 |
|---|---|---|---|
| contact_bonus active_rate | 11.8% | **0.0%** | iter_5 最终评估中着陆奖励完全死亡 |
| contact_bonus ep_sum | 2980 | 0 | iter_2 的 per-step 100 是唯一成功激活 contact 的方案 |
| goal_proximity ep_sum | -7.37 | **-36.24** | 距离惩罚累积 5 倍，因为 agent 在远处徘徊 938 步 |
| goal_proximity magnitude_share | 0.2% | **82.2%** | iter_5 中距离惩罚成为绝对主导信号 |
| velocity_penalty ep_sum | -4.60 | -5.80 | 大致相当 |
| len | 253 (全部终止) | 938 (75%截断) | iter_5 agent 不尝试着陆，在远处磨蹭到超时 |

**Checkpoint 趋势的关键线索 (iter_5)**：
- cp60%: contact_bonus=458.75, score=**+15.3** — agent 曾找到着陆！
- cp80%→cp100%: contact_bonus=0, score 从 -37.5 跌到 -92.9 — **学会了着陆然后又忘掉了**

**组件历史**：
- contact_bonus 只在 iter_2 真正激活过（per-step 100），其余所有迭代（iter_1/3/4/5）均接近死亡
- goal_proximity 在 iter_2 中微不足道（-7.37），在 iter_5 中暴涨到 -36.24（因为 episode 变长 3.7 倍）

### Behavior Diagnosis

当前策略学到的行为是：**「远离目标、低速徘徊」**。agent 发现只要保持低速（velocity_penalty 小）并远离目标（gate 使 velocity_penalty 更小），就能最小化惩罚。但由于 goal_proximity 始终为负，938 步累积了 -36 的巨额距离惩罚。contact_bonus 在训练中期曾激活（cp60%=458.75），说明 agent 有能力着陆，但 reward landscape 存在根本缺陷导致它「学会又忘掉」——**因为所有非着陆步都是纯惩罚，着陆本身的正信号（25/step）不足以在 PPO 的价值估计中形成稳定吸引子**。

**根因**：`goal_proximity = -w_dist * dist_nxt` 始终为负。当 contact_bonus 不够大时（iter_5 的 25 vs iter_2 的 100），整个 reward 空间是纯惩罚的，agent 的最优策略变成「别动、别靠近、别冒险」。

### 干预方案

**L2 修改**：将 `goal_proximity` 从「始终为负的距离惩罚」改为「基于势能的进度奖励」——接近目标时给正奖励，远离时给负奖励。

公式变换：`dist_cur - dist_nxt`（正值=接近，负值=远离）。这确保每一步朝目标移动都是净收益，agent 不再被「存在即惩罚」的困境压制。

同时保持 `contact_bonus` 为 per-step 模式（iter_2 唯一证明有效的形态），权重设为 80（接近 iter_2 的 100 但略低，因为进度奖励已提供正向引导）。

预期效果：
- 接近目标 → 持续正奖励 → agent 有动机探索目标区域
- 到达目标附近 + 偶然触地 → contact_bonus 80/step 形成悬崖式正反馈
- 着陆后 progress≈0，contact≈80/step → 净收益 ~80/step，明确优于徘徊

---

## 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v6: potential-based progress reward + per-step contact bonus (iter_2 skeleton)

    Core diagnosis from v5 failure:
    - v5's always-negative distance penalty (goal_proximity=-36.24, 82.2% share)
      created a pure-penalty landscape. Agent learned "hover far away, do nothing"
      because every step toward the target was punished.
    - contact_bonus (25/step) was too small to overcome the penalty gradient;
      agent found landing at cp60% but then abandoned it.
    - Only iter_2's per-step contact_bonus (100) ever successfully activated
      landing behavior. All attempts to "fix" the 99.6% dominance killed it.

    Changes from v5:
    1. goal_proximity: switched from distance penalty (-w*dist) to potential-based
       progress reward: w_pot * (dist_cur - dist_nxt). Positive when approaching,
       negative when retreating. This ensures every step toward target is net-positive
       regardless of contact, restoring exploration gradient.
    2. contact_bonus: per-step weight raised 25→80 — close to iter_2's proven 100,
       but slightly moderated since progress reward now also provides positive signal.
    3. velocity_penalty: slightly reduced 0.4→0.3 — progress reward already encodes
       a soft speed incentive (approaching requires some velocity).

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

    # --- 1. goal_proximity: potential-based progress reward ---
    # Rewards reduction in Euclidean distance to target (0,0).
    # Positive when approaching, negative when retreating.
    # Unlike distance penalty (always negative), this makes every approach step
    # net-beneficial, restoring the exploration gradient toward the target.
    # Weight 2.0 means approaching 0.1 units/step yields +0.2 reward.
    w_pot = 2.0
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = w_pot * (dist_cur - dist_nxt)

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Dampens high-speed approaches near the ground. Gate keeps far-field
    # maneuvering unpenalized so progress reward can drive initial approach.
    w_v = 0.3
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    # Weights unchanged — these have been stable across all iterations.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: per-step reward for safe pad contact ---
    # Per-step design (iter_2 proven): rewards every step with both legs on pad
    # near target. Creates natural curriculum from first touch → stable settle.
    # Weight 80: close to iter_2's 100 (which worked), moderated because progress
    # reward now provides positive approach gradient.
    w_contact = 80.0
    near_target = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    contact_bonus = w_contact if (near_target and both_legs) else 0.0

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