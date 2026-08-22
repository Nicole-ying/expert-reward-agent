# Response Record

Now I have all the data I need. Let me compile the diagnosis.

---

## 诊断报告

### evidence

**关键数据（iter_2 final-policy）：**
- Score = 79.8，离目标 200 差距大。terminated=20/20, len=315.8 — agent 能着陆但质量不佳。
- `contact_bonus`：ep_sum=441.8，magnitude_share=**92.2%**，但 active_rate 仅 5.7%。意味着 agent 仅在 ~18 步内吃到该奖励，却贡献了 92% 的 reward 量级。一个稀疏事件完全劫持了学习信号。
- `landing_approach`：ep_sum=17.2，magnitude_share=3.6% — 本来是连续引导信号，被 contact 碾压。
- `goal_proximity`：ep_sum=-13.3，始终为负 — 距离惩罚累积超过 approach bonus。

**Checkpoint 趋势揭示 exploit：**

| 指标 | cp40% | cp50% | cp70% | cp90% | cp100% | 趋势 |
|---|---|---|---|---|---|---|
| score | -28 | 12 | 98 | **102** | **68** | 峰值后下跌 33 分 |
| contact_bonus | 1844 | **1990** | 985 | 1065 | 630 | cp40% 暴涨 9x，然后衰减 |
| landing_approach | **55.6** | 35.3 | 26.4 | 16.8 | 16.3 | 持续萎缩 70% |
| velocity_penalty | -4.8 | -4.4 | -3.1 | -3.4 | **-6.2** | cp100% 突然恶化 |

- **contact_bonus 在 cp40% 骤升 9 倍**（199→1844），而 landing_approach 同时从 55.6 跌至 35.3。agent 发现了 contact 的巨大奖励后，放弃了 landing_approach 所鼓励的"优雅接近"，转为直奔接触。
- **score 在 cp90% 达峰后跌至 cp100%**：agent 在训练末期过度优化 contact_bonus，损害了着陆质量。velocity_penalty 在 cp100% 反弹至 -6.2（全程最高），暗示末期的着陆变得更粗暴（高速撞击式接触）。

**iter_1→iter_2 对比：**
- iter_1 完全没有 contact（active_rate=0%），score=-32，agent 从不终止。
- iter_2 引入 contact_bonus 后立刻学会着陆，但 contact_bonus=25.0 的权重大到失控。

### behavior_diagnosis

当前策略已学会 **"只要能碰到平台就算赢"** 的行为模式。agent 从 cp40% 开始发现 contact_bonus 是一座金矿（每步 25 点），于是全力追求接触，忽视着陆质量：
- landing_approach 萎缩说明 agent 不再关心"低速、居中、直立地接近"
- velocity_penalty 末期恶化说明 agent 以更高速度撞击平台
- score 末期下跌说明这种行为在 evaluation 中受到惩罚（可能是高速撞击导致 crash，或姿态不稳导致单脚着陆）

**核心问题：contact_bonus 是 per-step 累积的，导致一个短期事件产生了不可控的量级。必须改为一次性奖励，并大幅降低权重。同时需要加强连续引导信号（landing_approach、goal_proximity）来填补 contact 留下的梯度真空。**

### 干预方案

1. **contact_bonus → 一次性（transition-based）**：用 `obs` vs `next_obs` 的 contact 状态变化检测"首次接触"，只奖励一次。权重从 25→8，外加速度门控（高速接触不给满额）。
2. **landing_approach 权重翻倍**：0.3→0.6，温度参数稍放宽，让它成为主要的连续正向信号。
3. **goal_proximity approach 权重翻倍**：0.4→0.8，增强"靠近即奖励"的梯度。
4. **velocity_damping 加强**：w_v 0.15→0.20，gate 更陡，在近目标区更强制减速。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: one-shot contact bonus (transition-based) + stronger shaping signals

    Core fix: contact_bonus was per-step with w=25, accumulating to 92% of reward.
    Now it fires ONCE on the transition from "no contact" → "contact", with a
    velocity gate that penalises high-speed touchdowns. The continuous shaping
    signals (landing_approach, goal_proximity) are strengthened to fill the gradient
    that contact_bonus previously monopolised.
    """

    # --- unpack ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_cur, right_cont_cur = obs[6], obs[7]
    left_cont_nxt, right_cont_nxt = next_obs[6], next_obs[7]

    # --- helpers ---
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2

    # ================================================================
    # 1. goal_proximity — distance penalty + approach incentive
    #    w_approach doubled (0.4 → 0.8) to give stronger gradient
    # ================================================================
    w_dist = 0.08
    proximity_penalty = -w_dist * dist_nxt

    w_approach = 0.8
    delta_dist = dist_cur - dist_nxt          # positive = getting closer
    approach_bonus = w_approach * max(0.0, delta_dist)

    goal_proximity = proximity_penalty + approach_bonus

    # ================================================================
    # 2. velocity_damping — penalise speed, stronger near the pad
    #    w_v increased 0.15→0.20, k_gate 2.0→3.0 for sharper gating
    # ================================================================
    w_v = 0.20
    k_gate = 3.0
    gate = 1.0 / (1.0 + k_gate * dist_nxt)    # ~1 when close, ~0 when far
    velocity_penalty = -w_v * gate * speed_sq

    # ================================================================
    # 3. orientation_penalty — unchanged weight, effective
    # ================================================================
    w_ang = 0.03
    w_angvel = 0.005
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ================================================================
    # 4. landing_approach — weight doubled (0.3 → 0.6),
    #    temperature parameters slightly widened for broader basin
    # ================================================================
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.30)   # was 0.25
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.40)   # was 0.35
    slow     = 2.718281828 ** (-speed_sq / 1.0)      # was 0.8
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.35) # was 0.3
    landing_approach = 0.6 * near_y * near_x * slow * upright

    # ================================================================
    # 5. contact_bonus — ONE-TIME transition bonus (not per-step!)
    #    Detects the moment both legs first touch down.
    #    Velocity gate ensures high-speed impacts get less credit.
    #    w_contact = 8.0 (was 25.0 per-step → now ~8 once per episode).
    # ================================================================
    w_contact = 8.0
    had_contact = left_cont_cur * right_cont_cur       # 0 or 1, were we already touching?
    has_contact = left_cont_nxt * right_cont_nxt       # 0 or 1, are we touching now?
    new_contact = has_contact * (1.0 - had_contact)    # 1 ONLY on the transition step

    spatial_gate = 2.718281828 ** (-(x_nxt ** 2 + y_nxt ** 2) / 0.5)
    soft_gate    = 2.718281828 ** (-speed_sq / 0.5)    # high speed → bonus shrinks
    contact_bonus = w_contact * new_contact * spatial_gate * soft_gate

    # ================================================================
    # 6. fuel_efficiency — unchanged, tiny regularisation
    # ================================================================
    w_fuel = 0.003
    fuel_penalty = -w_fuel * (0.0 if action == 0 else 1.0)

    # ================================================================
    # assemble
    # ================================================================
    total_reward = (
        goal_proximity
        + velocity_penalty
        + orientation_penalty
        + landing_approach
        + contact_bonus
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'contact_bonus': contact_bonus,
        'fuel_penalty': fuel_penalty,
    }

    return total_reward, components
```
