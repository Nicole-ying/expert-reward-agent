## 诊断

### 1. Evidence

- **score=59.61, len=623.4**：episode 偏长（623 步），说明 agent 在着陆后没有快速 settle，可能在垫子上弹跳或维持不稳定接触。
- **contact_bonus 主导**：episode_sum_mean=391.13，magnitude_share=90.5%，active_rate=16.8%。在 623 步中约 105 步触发，每步 ~3.75，累积巨大。
- **landing_progress 微弱**：episode_sum_mean=0.80，虽 100% 触发但每步贡献 ~0.0013，几乎不提供可区分梯度。
- **persistent state reward**：`contact_bonus = 5.0 * both_contact * phi` 是持续状态奖励——每步双脚接触都发奖。这激励 agent 维持接触而非快速稳定下来（body_not_awake_or_settled 会终止 episode）。

### 2. Diagnosis Dimension

**persistent_to_transition_event**：contact_bonus 是对"处于接触状态"的持续奖励，而非对"完成接触动作"的事件奖励。agent 可以通过在垫子上弹跳或长时间不稳定接触来反复获利，与"尽快稳定着陆"的目标冲突。

### 3. Selected Transformation

`persistent_to_transition_event` — 将 `contact_bonus` 从持续状态奖励改为转移事件奖励：仅在从「双脚未接触」到「双脚接触」的转移步发放一次性 bonus。

匹配理由：contact_bonus 的 active_rate=16.8% 和 magnitude_share=90.5% 是典型的 persistent reward farming 特征；agent 在接触状态停留越久获利越多，与 settled 终止形成张力。

### 4. Falsifiable Hypothesis

将 contact_bonus 改为 transition event 后，agent 不再从「维持接触」中获利，唯一获利路径是：飞到垫子 → 双脚触地（获一次性奖励）→ 通过 potential shaping 稳定下来。这应使 episode 缩短（更快 settle），contact_bonus 的 magnitude_share 下降，但 landing_progress 的引导作用更突出。

### 5. Expected Next Round

- **score**：可能短期略降（因为 contact_bonus 总量暴跌），但 episode_length 应缩短
- **episode_length**：应下降（不再激励弹跳/维持接触）
- **contact_bonus**：magnitude_share 大幅下降，episode_sum_mean 降至每次接触事件 ~3-5 的量级
- **landing_progress**：相对份额上升，但绝对值可能仍小

### 6. Main Risk

transition event 仍可被弹跳利用（弹起→再次触地→再次获奖），但利用难度远高于原地驻留。若下轮出现大量短周期弹跳，需加 cooldown 或改为仅首次接触事件。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励函数：势能差塑形 + 接触转移事件奖励。

    核心变换 (persistent_to_transition_event)：
    将 contact_bonus 从「每步双脚接触都发奖」改为「仅在双脚从未接触到接触的转移步发奖」，
    消除 agent 在垫上驻留/弹跳刷分的动机。

    Φ(s) = 1 / (1 + cost)，cost 加权综合距离/速度/姿态。
    """
    # ── 当前状态 ──
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left = next_obs[6]
    right = next_obs[7]

    # ── 上一步状态 ──
    px = obs[0]
    py = obs[1]
    pvx = obs[2]
    pvy = obs[3]
    p_angle = obs[4]
    p_angvel = obs[5]
    p_left = obs[6]
    p_right = obs[7]

    # ── 当前代价 ──
    dist = (x * x + y * y) ** 0.5
    speed = (vx * vx + vy * vy) ** 0.5
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    cost = (
        0.5 * dist +
        0.5 * speed +
        2.0 * abs_angle +
        1.0 * abs_angvel
    )

    # ── 上一步代价 ──
    prev_dist = (px * px + py * py) ** 0.5
    prev_speed = (pvx * pvx + pvy * pvy) ** 0.5
    prev_abs_angle = abs(p_angle)
    prev_abs_angvel = abs(p_angvel)

    prev_cost = (
        0.5 * prev_dist +
        0.5 * prev_speed +
        2.0 * prev_abs_angle +
        1.0 * prev_abs_angvel
    )

    # ── 势能函数：Φ(s) = 1/(1+cost)，有界 ∈ (0, 1] ──
    phi = 1.0 / (1.0 + cost)
    prev_phi = 1.0 / (1.0 + prev_cost)

    # ── 势能差塑形：Φ(next) - Φ(prev) ──
    progress_scale = 10.0
    landing_progress = progress_scale * (phi - prev_phi)

    # ── 接触转移事件奖励：仅在双脚从「未接触」→「接触」的转移步发奖 ──
    prev_both = p_left * p_right      # 上一步双脚接触标志
    curr_both = left * right          # 当前步双脚接触标志
    contact_event = float(curr_both > 0.5 and prev_both < 0.5)  # 转移事件

    contact_scale = 5.0
    contact_bonus = contact_scale * contact_event * phi

    total_reward = landing_progress + contact_bonus

    components = {
        "landing_progress": landing_progress,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```