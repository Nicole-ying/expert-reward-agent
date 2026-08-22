# Response Record

Now I have the full picture. Let me compile the diagnosis and intervention.

---

## 1. 诊断报告

**evidence**：
- 得分 -32.35，距目标 200 差距巨大；terminated=0/20，全部 truncate 在 1000 步 — agent 从未成功着陆。
- contact_bonus 完全死亡：active_rate=0.0%，ep_sum=0.0。整个训练过程中唯一一次激活在 cp10%（active_rate=0.8%, ep_sum=5.162），当时总分高达 38.8，随后完全消失。
- velocity_penalty 过于微弱：ep_sum=-0.486，magnitude_share 仅 5.7%。agent 接近地面时没有足够梯度去减速。
- goal_proximity (+3.76) 和 orientation_penalty (-3.96) 几乎完全对冲，agent 在每一步的净信号接近零。
- Checkpoint 趋势：cp10% 得分 38.8（contact_bonus 曾激活），cp20% 骤降至 -27.4（contact_bonus 归零），之后再也没恢复。说明早期探索曾偶然触达着陆条件，但因信号太稀疏，策略漂移后无法重新发现。

**behavior_diagnosis**：
Agent 学会了向目标 (0,0) 靠近并保持姿态，但**从不减速着陆**。它在空中持续飞行 1000 步直到 truncate。velocity_penalty 使用的 Manhattan 距离门控衰减太慢（在距离 5 处 gate 仍有 ~0.17），但权重又太低（w_v=0.05），导致无论远近速度惩罚都不足以驱动减速行为。contact_bonus 的条件（双腿同时触地 + 近靶心）过于苛刻且稀疏，agent 在整个训练中几乎从未触发，无法提供着陆梯度。

**signal_completeness**：
四个必要职责中三个有信号但引导失效：
- goal_proximity ✅ 有效但弱（delta 型，接近后梯度消失）
- orientation_penalty ✅ 有效但始终在线，无接近地面时放宽的机制
- velocity_damping ❌ 形式合理但权重太低 + 门控不精准：高度方向（y）才是着陆关键维度，当前 Manhattan 门控把 x 也等权混入，稀释了"近地减速"的语义
- contact_bonus ❌ 死组件：稀疏二值信号，无梯度

**selected_level**：**L2** — 修改 velocity_penalty 的数学形态，将门控函数从 Manhattan 距离改为纯高度指数衰减，同时提升权重。

**selected_intervention**：
修改 `velocity_penalty` 组件：
- 门控函数：`1/(1 + |x|+|y|)` → `exp(-y_nxt / h_scale)`，使惩罚仅在地面附近（y 小）时生效
- 权重：`w_v` 从 0.05 → 0.5（10x），使近地速度惩罚与 goal_proximity/orientation_penalty 量级匹配
- h_scale=1.0：在 y=0 处 gate≈1.0，y=1 处 gate≈0.37，y=3 处 gate≈0.05，远距惩罚几乎消失

**falsifiable_hypothesis**：
更强的近地速度惩罚将驱动 agent 在接近地面时主动减速（使用主引擎），从而创造软着陆条件。一旦 agent 学会减速接近 (0,0)，contact_bonus 的苛刻条件（双腿触地 + 近靶心）变得可达，contact_bonus 将从死组件恢复激活，提供最终着陆奖励。预期形成正向飞轮：减速→接近→接触→高奖励→强化减速行为。

**expected_next_round**：
- velocity_penalty 的 ep_sum 应显著增大（从 -0.49 → 约 -3 到 -8），magnitude_share 上升到 20-40%
- contact_bonus 的 active_rate 应突破 0%，至少达到 1-5%
- 得分应改善（目标：从 -32 提升到 -10 以上）
- 可能出现少量 early terminal（成功着陆导致 episode 提前结束）
- 如果 velocity_penalty 的 magnitude_share > 60%，说明权重过高抑制了接近行为，下轮需回调

**main_risk**：
Agent 可能学会"永不下降"来规避近地速度惩罚 — 如果 goal_proximity 的 y 方向拉力弱于 velocity_penalty 的近地推力，agent 会悬停在高空。缓解因素：goal_proximity 的 w_y=2.0 提供了较强的下降拉力，且远距 gate≈0 不惩罚速度，agent 可以高速下降到中等高度再减速。

---

## 2. 完整 Python 代码

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
