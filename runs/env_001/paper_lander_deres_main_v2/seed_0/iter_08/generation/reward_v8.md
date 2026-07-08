## 诊断分析

### 1. Evidence
- **score=91.87**（best so far），但 **len=978.8，truncated=19/20**：几乎所有 episode 都跑满步数被截断，只有 1/20 真正终止。
- **contact_bonus**：episode_sum_mean=639.35，signed_share=70.3%，active_rate=**13.7%**。978 步 × 13.7% ≈ 134 次接触转移事件。这说明 agent 在反复弹跳——每次弹起再落下都触发一次 contact_event。
- **landing_progress**：active_rate=100%，但 signed_share 仅 0.5%。势能差信号在全程提供反馈但被 contact_bonus 淹没。
- 得分虽好，但行为模式显示 **contact_bonus 驱动了弹跳刷分**，而非稳定着陆后停留。

### 2. Diagnosis Dimension
当前缺少的性质：**抗利用性（anti-hacking）**。contact_event 虽已从持续奖励改为转移事件，但 agent 通过反复弹跳（加速→撞垫→弹起→再落下）持续触发转移事件，形成新的 exploit。

### 3. Selected Transformation
**state_to_improvement / quality-gated transition**：在 contact_event 基础上叠加 **速度门控（speed gate）**——仅当接触转移发生且着陆速度足够低（软着陆）时才发奖。这属于对 contact_bonus 单个组件的数学形式修改，符合"每次只改一个组件"原则。

### 4. Falsifiable Hypothesis
如果 agent 因为弹跳刷分，那么对高速接触不给奖励后，弹跳不再累积收益，agent 应学会减速后轻柔着陆并保持接触。预期：contact_bonus 的 active_rate 和 episode_sum_mean 下降，同时 episode_length 缩短（更多 episode 真正终止而非截断），score 维持或提升。

### 5. Expected Next Round
- contact_bonus episode_sum_mean 下降，active_rate 略降（只保留真正的软着陆事件）
- episode_length 缩短，terminated 比例上升
- score 至少不下降，理想情况下因更高效着陆而提升

### 6. Main Risk
速度门控可能使 contact_bonus 过于稀疏，尤其早期探索阶段 agent 难以同时满足"双脚接触转移 + 低速"两个条件。如果下一轮 active_rate 暴跌至接近 0 且 score 下降，需要放宽速度阈值或改为连续 softness 而非硬门控。

---

现在生成完整代码。核心改动：在 `contact_bonus` 中乘入 `softness = max(0, 1 - speed/safe_speed)`，只对低速软着陆发奖。其余结构保持不变。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励函数：势能差塑形 + 速度门控接触转移事件奖励。

    核心变换 (quality-gated transition)：
    将 contact_bonus 从「任何双脚接触转移都发奖」改为「仅低速软着陆的接触转移发奖」，
    消除 agent 通过高速弹跳反复触发接触事件刷分的动机。

    Φ(s) = 1 / (1 + cost)，cost 加权综合距离/速度/姿态。
    """
    # ── 当前状态 ──
    x = float(next_obs[0])
    y = float(next_obs[1])
    vx = float(next_obs[2])
    vy = float(next_obs[3])
    angle = float(next_obs[4])
    angvel = float(next_obs[5])
    left = float(next_obs[6])
    right = float(next_obs[7])

    # ── 上一步状态 ──
    px = float(obs[0])
    py = float(obs[1])
    pvx = float(obs[2])
    pvy = float(obs[3])
    p_angle = float(obs[4])
    p_angvel = float(obs[5])
    p_left = float(obs[6])
    p_right = float(obs[7])

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

    # ── 接触转移事件 + 速度门控 ──
    # 双脚接触转移：从未接触 → 接触
    prev_both = 1.0 if (p_left > 0.5 and p_right > 0.5) else 0.0
    curr_both = 1.0 if (left > 0.5 and right > 0.5) else 0.0
    contact_event = 1.0 if (curr_both > 0.5 and prev_both < 0.5) else 0.0

    # 速度门控：仅低速软着陆时发奖，高速弹跳不给奖励
    safe_speed = 1.5  # 安全着陆速度阈值
    softness = max(0.0, 1.0 - speed / safe_speed)

    contact_scale = 5.0
    contact_bonus = contact_scale * contact_event * phi * softness

    total_reward = landing_progress + contact_bonus

    components = {
        "landing_progress": landing_progress,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```