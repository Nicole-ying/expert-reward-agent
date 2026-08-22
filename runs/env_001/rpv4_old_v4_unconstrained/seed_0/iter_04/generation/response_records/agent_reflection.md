# Response Record

现在我有了完整的图景。让我整理诊断和方案。

---

## 诊断报告

### Evidence

| 证据来源 | 关键数据 |
|---|---|
| **iter_3 score** | -166.29（目标 200），20/20 truncated（全部跑满 1000 步，无终止） |
| **iter_3 score 轨迹** | cp10%=-2591 → cp80%=**-51（峰值）** → cp100%=**-158.7（退化）** |
| **proximity_bonus** | ep_sum=+768.78，magnitude_share=**94.3%**，全程 100% active |
| **contact_reward** | 0.0，全部 checkpoint 为 0 — agent **从未触碰地面** |
| **landing_bonus** | 0.0，全部 checkpoint 为 0 — agent **从未尝试着陆** |
| **vel_damping active_rate** | 从 cp10% 的 43.5% 跌至 cp100% 的 3.7% |
| **iter_2（对照）** | len=68.45，全部 early terminal — agent 快速 crash，但至少接触了地面（contact 3.1% active） |
| **iter_1（对照）** | len=303.35，负向距离+速度阻尼主导，同样 crash |

### Behavior Diagnosis

**经典 reward hacking**：agent 发现 `proximity_bonus = 2.0/(1+dist)` 是一个永不枯竭的收入源。在 dist≈0 时每步 +2.0，1000 步可累计 +2000，远超 landing_bonus 的一次性 +10。agent 的最优策略是：飞到目标垫附近，然后**悬停 1000 步吃满 proximity 收益，绝不落地**。

证据链：
1. Score 在 cp80% 达到峰值 -51（学会了接近垫子），随后退化到 -158（学会了悬停剥削）
2. vel_damping active_rate 从 43.5% 跌至 3.7%——agent 学会停在刚好不触发速度阻尼的距离
3. contact/landing 全程为 0——agent 主动避免接触地面
4. 全部 20 集 truncated（跑满 1000 步），无任何终止

**根因**：per-step proximity 奖励是正的、无上限的、且不需要完成任务即可获取。这比 sparse landing bonus 更有吸引力。

### 干预策略

**Level 3 — 结构性重写**：移除 per-step proximity_bonus，代之以 progress-based 导航信号（delta-distance），让 landing_bonus 成为唯一的大额正奖励来源。contact_reward 作为中间引导，鼓励探索地面接触。

核心原则：
- **不做任何 per-step 位置奖励** — 只奖励"靠近"的动作，不奖励"在附近"的状态
- **landing_bonus 足够大**（+50）— 让着陆成为最优选择
- **contact_reward 有意义** — 每只脚 +0.5，鼓励尝试接触
- **action 0 中性** — 滑翔不奖励也不惩罚，避免"什么都不做"也能得分

---

## v4 奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: Eliminate per-step proximity exploitation.

    Core principles:
    - NO per-step location bonus — only delta-distance progress shapes navigation
    - Landing bonus is the LARGEST positive signal, making landing the optimal choice
    - Contact reward encourages ground interaction for discoverability
    - Velocity damping only near pad, for soft landing guidance
    - Action 0 (glide) is neutral — doing nothing forever earns nothing
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    x_prev = obs[0]
    y_prev = obs[1]

    # ── 1. Progress: delta-distance (THE navigation signal) ──
    # Positive when approaching pad, negative when retreating.
    # No per-step reward for just *being* near the pad.
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    progress_reward = 1.5 * (dist_prev - dist_curr)

    # ── 2. Orientation penalty (moderate) ──
    orientation_penalty = -0.1 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 3. Velocity damping near pad ──
    # Only penalizes excess speed when close to pad
    speed = (vx ** 2 + vy ** 2) ** 0.5
    proximity_gate = 2.718281828 ** (- (dist_curr ** 2) / 0.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * proximity_gate * excess_speed

    # ── 4. Contact reward: any foot on ground is progress ──
    # Per-step reward encourages the agent to descend and try touching
    contact_sum = left_contact + right_contact
    contact_reward = 0.5 * contact_sum

    # ── 5. Landing success bonus: LARGE sparse reward ──
    # This MUST be the most attractive positive signal
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist_curr < 0.6)
    upright = (abs(angle) < 0.35)
    slow = (speed < 0.5)
    is_good_landing = both_on_ground and near_pad and upright and slow
    landing_bonus = 50.0 if is_good_landing else 0.0

    # ── 6. Action efficiency ──
    # Glide (action 0) is neutral — no free lunch for doing nothing
    if action == 0:
        action_reward = 0.0
    elif action == 2:          # main engine — highest cost
        action_reward = -0.04
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.02

    # ── Aggregate ──
    total_reward = (
        progress_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + landing_bonus
        + action_reward
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "action_reward": action_reward,
    }
    return float(total_reward), components
```

### 与 v3 的关键差异

| 维度 | v3（问题） | v4（修正） |
|---|---|---|
| **导航信号** | `2.0/(1+dist)` per-step，可剥削 | `1.5*(dist_prev - dist_curr)`，只奖励接近动作 |
| **悬停收益** | dist≈0 时 +2.0/步 → 1000步≈+2000 | dist不变时 ≈0/步 → 悬停无收益 |
| **着陆激励** | +10 一次性，远小于悬停累计收益 | +50 一次性，是唯一大额正奖励 |
| **滑翔** | +0.02/步（可累积） | 0.0（中性，不做无意义滑翔） |
| **接触引导** | 1.0/脚 | 0.5/脚（适度，防止成为新剥削源） |
