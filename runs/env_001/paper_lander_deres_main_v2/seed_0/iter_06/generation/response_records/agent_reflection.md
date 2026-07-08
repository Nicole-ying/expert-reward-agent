# Response Record

## 诊断

### 1. 这个 agent 发生了什么？

**episode 很长（1000步，全部截断），得分仍差（-11.99），但比之前（-108）大幅改善。** 

关键证据：
- `landing_quality` 100% 活跃，episode_sum_mean = 749 — agent 每步都在累积正向奖励
- `contact_bonus` 0% 活跃 — agent 从未双脚同时接触着陆垫
- 外部 score = -11.99，远好于之前的 -108，但仍为负值

agent 学会了在目标附近保持低距离、低速度、良好姿态来持续收获 `landing_quality` 奖励，但没有学会最后一步：双脚着陆。这是典型的 **persistent proxy farming**（持续代理奖励农耕）：占据"好状态"就能持续获奖，无需完成任务。

### 2. 哪个组件最值得干预？

`landing_quality`。它以 `exp(-cost)` 的数学形式每步发放正奖励，agent 只需维持状态就能累积大量奖励。`contact_bonus` 虽然语义正确但过于稀疏（0% 触发），无法单独引导着陆。

### 3. 我之前改了什么？

Iter 5 从 `progress_delta + soft_landing_proxy + stability_penalty` 改为 `landing_quality + contact_bonus`，得分从 -108 跳到 -11.99。方向对（稠密质量信号有效），但数学形态错（持续状态奖励 → 农耕）。

---

## 修改方案

### evidence
- `landing_quality` episode_sum_mean=749, active_rate=100%, magnitude_share=100%
- `contact_bonus` 完全未触发
- episode 全部跑满 1000 步截断，无早期终止
- 外部 score = -11.99

### diagnosis_dimension
**persistent proxy farming** — 持续状态奖励使 agent 通过"停在好状态"而非"完成任务"来最大化奖励。

### selected_transformation
**`state_to_improvement`**（persistent → potential-based progress delta）：将 `quality(s)` 从每步发放改为势能差 `Φ(s_next) - Φ(s)`，agent 只为改善状态获得奖励，而非占据状态。

### falsifiable_hypothesis
当 agent 无法通过维持状态获得奖励时，它必须通过改善状态来获得正反馈——最终必须完成着陆才能停止"损失"（或获得最大正奖励）。progress delta 提供了朝向目标的每步梯度，同时消除农耕动机。

### expected_next_round
- `landing_progress` 的 episode_sum_mean 应远小于 749（不再累积）
- `contact_bonus` active_rate 应从 0% 上升
- episode_length 应从 1000 开始下降（agent 学会更快着陆）
- 外部 score 应改善

### main_risk
纯 progress delta 在到达最优状态后趋近于 0，agent 可能"着陆后离开"以获取更多 progress。`contact_bonus` 提供维持着陆的微弱激励，但如果系数太小可能不足以对抗此风险。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励函数：势能差塑形 + 接触事件奖励。
    
    核心变换：将 persistent quality(s) 替换为 potential_delta = Φ(next) - Φ(prev)，
    agent 只为改善状态获得正奖励，消除"停留农耕"。
    
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

    # ── 势能差塑形：Φ(next) - Φ(prev)，乘以缩放因子 ──
    progress_scale = 10.0
    landing_progress = progress_scale * (phi - prev_phi)

    # ── 接触事件奖励：双脚着陆 × 当前状态质量 ──
    both_contact = left * right  # 1.0 仅当双脚同时接触
    contact_scale = 5.0
    contact_bonus = contact_scale * both_contact * phi

    total_reward = landing_progress + contact_bonus

    components = {
        "landing_progress": landing_progress,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```
