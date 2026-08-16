Now I have a complete picture. Let me synthesize all findings into the diagnosis.

---

## 1. 诊断报告

### evidence

**全局模式 — contact_bonus 的「死-活交替」彩票：**

| iter | contact_bonus ep_sum | contact_bonus active_rate | score | terminated |
|------|---------------------|--------------------------|-------|------------|
| 1 | 0.000 | 0.0% (dead) | -32.35 | 0/20 |
| 2 | 154.465 | 2.8% (alive) | 107.13 | 11/20 |
| 3 | 0.000 | 0.0% (dead) | -26.17 | 0/20 |
| 4 | 114.083 | 4.9% (alive) | **183.52** | 19/20 |
| 5 | 0.000 | 0.0% (dead) | -15.01 | 0/20 |

contact_bonus 公式从未变过（始终是 `w * y_close * x_close * left * right`），只有权重在 10→4→5 之间调整。当 agent 偶然发现着陆时（iter 2, 4），contact_bonus 占据 86-92% magnitude_share，整个训练被这个稀疏信号主导。当 agent 没发现着陆时（iter 1, 3, 5），contact_bonus 完全死亡，得分停留在 -15 到 -32。

**iter_5 checkpoint 的关键证据：**
- cp80% 时 contact_bonus 短暂激活（125.97, active_rate 5.3%），得分冲到 193.2——几乎追平 iter_4 最佳
- cp90% 和 cp100% 时 contact_bonus 归零，得分回落到 -10~-20
- 这说明 agent **有能力着陆**，但策略不稳定——一旦探索噪声使策略偏离着陆轨道，稀疏的 contact_bonus 消失，没有足够的连续梯度把策略拉回来

**descent_incentive 太弱：**
- iter_4: ep_sum=0.779 (72.4% active)，iter_5: ep_sum=0.624 (66.1% active)
- 相对于 contact_bonus 的 114，descent_incentive 的 ~0.6-0.8 微不足道——它没有足够的力量独立引导着陆
- near_ground gate 在 y=0.8 处截断，agent 在 y>0.8 时完全得不到下降激励

**iter_5 的 L1 修改（contact_bonus w 4→5）完全失败：**
- 上一轮诊断认为「差距仅 8.2%，只需微调 contact_bonus 权重」
- 但实际结果：contact_bonus 完全死亡，得分从 183.52 暴跌到 -15.01
- 根因：contact_bonus 是稀疏彩票，调权重不改变稀疏性。权重 4 或 5 对「是否能被发现」没有影响——agent 要么偶然发现（得分好），要么没有（得分差）

### behavior_diagnosis

策略在两个极端之间摇摆：要么偶然发现着陆并获得巨额 contact_bonus 奖励（iter 2, 4, iter_5 cp80%），要么完全不会着陆（iter 1, 3, 5 最终）。**着陆行为本身是可学习的**（cp80% 证明），但策略不稳定——descent_incentive 太弱，无法在 contact_bonus 消失时维持下降梯度。agent 一旦偏离着陆轨道，就没有足够的连续信号引导它回来。

### signal_completeness

所有必要职责已覆盖（goal_proximity、velocity_penalty、orientation_penalty、descent_incentive、contact_bonus），但 **descent_incentive 的强度严重不足**。它的 near_ground gate（y<0.8）过于狭窄，且权重 2.0 太小，导致 per-step 信号仅为 ~0.0006——比 velocity_penalty（~-0.007）小一个数量级。

### selected_level

**L2** — 修改 descent_incentive 的数学形态（加宽 gate + 提权），同时将 contact_bonus 权重回退到 iter_4 验证过的 4.0。选择 L2 而非 L1 的理由：descent_incentive 的 gate 宽度（0.8）从根本上限制了它能作用的范围，仅调权重不足以弥补。选择 L2 而非 L3 的理由：整体框架在 iter_4 已验证有效（score 183.5），不需要换主信号。

### selected_intervention

**目标组件：descent_incentive**。具体修改：

1. **加宽 near_ground gate**：从 `1.0 - y/0.8`（y=0.8 截断）改为 `1.0 - y/1.5`（y=1.5 截断）。agent 从更高处就开始收到下降激励，覆盖更大的状态空间。
2. **提升权重**：`w_desc` 从 2.0 → 6.0。配合更宽的 gate，预期 descent_incentive ep_sum 从 ~0.6 提升到 3-6 范围，使其与 goal_proximity（~7.9）可比，能独立提供有意义的下降梯度。
3. **回退 contact_bonus 权重**：5.0 → 4.0（iter_4 验证值）。5.0 的改动是 iter_5 失败的诱因之一，回退是防御性的。

这三个修改服务于同一语义：「让着陆引导信号更可靠、更连续」，减少对稀疏 contact_bonus 彩票的依赖。

### falsifiable_hypothesis

如果 descent_incentive 足够强，agent 应该能**稳定发现着陆**——即使 contact_bonus 在训练早期还未激活，descent_incentive 提供的连续下降梯度也能引导 agent 进入着陆区。训练过程中 contact_bonus 不应再出现「全程死亡→短暂激活→消失」的模式，而应表现为：descent_incentive 先引导 agent 下降到着陆区，然后 contact_bonus 自然激活并稳定增长。

### expected_next_round

- **Score**：应回到正区间，目标 ≥ 150。如果能稳定着陆（terminated > 10/20），contact_bonus 将提供额外推力。
- **descent_incentive ep_sum**：应从 ~0.6 提升到 3-8，active_rate 从 66% 提升到 75-90%。
- **contact_bonus active_rate**：应从 0% 提升到 > 3%，ep_sum > 50。
- **velocity_penalty magnitude_share**：应从 35% 下降（因为正信号变强）。
- **Checkpoint 趋势**：不应再出现「cp80% 短暂峰值后崩塌」的模式，分数应在训练后期稳定或持续改善。

### main_risk

descent_incentive 过强可能导致 agent **在着陆区附近上下振荡**——下降获得 descent_incentive 奖励，上升获得 goal_proximity 的负 delta_y_sq 惩罚，但如果 net 仍为正，agent 可能学会「下降-弹起-再下降」的循环来刷奖励。不过 goal_proximity 的 squared-distance 形式对上升的惩罚较重（w_y=4.0），且 near_center gate 限制了水平偏离，这个风险可控。如果下一轮出现异常高的 descent_incentive magnitude_share（> 50%），需考虑加 oscillation 防护。

---

## 2. 完整 Python 代码

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