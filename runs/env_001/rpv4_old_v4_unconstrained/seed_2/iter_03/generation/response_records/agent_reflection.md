# Response Record

现在我对全局数据有了完整把握。让我系统地诊断。

---

## 诊断报告

### Evidence

| 信号 | 数据 |
|---|---|
| **contact_guidance** magnitude_share | **70.0%** — 仍在主导，但仅 26.7% active |
| **contact_guidance** checkpoint 轨迹 | 0 → 0 → 4.5 → **84.5** → **0** → 19.5 → 67.9 → 48.4 → **128.7** → 29.4（剧烈震荡！） |
| Overall score checkpoint | -107.9 → -359.9 → 24.7 → 191.1 → **-11.0** → 258.0 → ... （cp50% 崩溃） |
| **progress** signed_share | 仅 5.8%，ep_sum_mean=4.15，在 contact 面前太弱 |
| **velocity_management** | ep_sum=-13.8，98.3% active，持续负向拖累 |
| iter_1 → iter_2 趋势 | contact 从 99.2% → 70.0%，score 171→253，len 781→387 — 收紧 gate 有效但不够 |

### Behavior Diagnosis

**核心问题：contact_guidance 的硬截断造成奖励悬崖。** 当前 `proximity_ok = max(0, 1 - dist/0.3)` 在 dist=0.29 时给满、dist=0.31 时归零。这制造了一个极窄的"奖励峡谷"——agent 偶尔撞进去获得巨大 burst（cp90% 时 contact=128.7），但这个脆弱的策略在下一次策略更新中就被冲垮（cp50% 时 contact=0，score 从 191 暴跌到 -11）。Agent 在整个训练过程中反复在「偶然发现 contact 奖励 → 过度 exploit → 策略崩塌 → 重新探索」的循环中震荡。

**次要问题：velocity_management 的 flat speed penalty 无差别惩罚所有速度。** 当 agent 远离目标时需要高速接近，-0.08*speed_sq 却在全程惩罚。descent_penalty 对所有向下速度都惩罚，连安全着陆的下沉速度也罚，实际上在阻止着陆。

**策略行为模式：** Agent 学到的不是「导航到目标并软着陆」，而是「在 contact reward zone 附近试探，偶尔触发奖励 burst」。19/20 terminated 说明大部分 episode 能触发 settled，但 checkpoint 的不稳定性说明这个策略不鲁棒。

### 干预方案

1. **contact_guidance：悬崖 → 缓坡** — Gaussian 替代硬截断，提供连续的梯度引导
2. **velocity_management：无条件惩罚 → 距离自适应** — 远处允许高速，近处要求低速；下降惩罚仅针对过快下沉
3. **progress：3.0 → 5.0** — 强化导航主信号，与 contact 形成更好平衡
4. **contact 系数保持 0.3** — Gaussian 变平滑后有效范围扩大，不需要提系数

---

## 完整代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── Distance to target (origin at 0, 0) ──
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # ── 1. Progress: strong navigation signal ──
    # potential-based shaping — episode total = dist_start - dist_end
    progress = dist_prev - dist_next          # >0 when approaching target
    progress_reward = 5.0 * progress          # was 3.0, boost to compete with contact

    # ── 2. Adaptive velocity management ──
    vx = next_obs[2]
    vy = next_obs[3]
    speed_sq = vx ** 2 + vy ** 2
    speed = speed_sq ** 0.5

    # 2a. Distance-scaled target speed: fast OK when far, must slow down when close
    #     target_speed → 0.6 when dist is large, → 0 when dist → 0
    target_speed = 0.6 * (1.0 - 2.718281828 ** (-3.0 * dist_next))
    speed_excess = max(0.0, speed - target_speed)
    speed_penalty = -0.08 * speed_excess ** 2   # quadratic penalty on excess only

    # 2b. Safe-descent guidance: allow moderate downward speed, penalise only excess
    safe_descent_rate = 0.4                     # up to 0.4 downward is acceptable
    near_ground = 2.718281828 ** (-3.0 * dist_next)
    descent_excess = max(0.0, -vy - safe_descent_rate)
    descent_penalty = -0.6 * descent_excess * near_ground

    velocity_management = speed_penalty + descent_penalty

    # ── 3. Orientation stabilisation ──
    orientation_penalty = -0.15 * abs(next_obs[4]) - 0.05 * (next_obs[5] ** 2)

    # ── 4. Contact guidance: smooth Gaussian — no more cliff ──
    contact_val = float(next_obs[6] + next_obs[7])   # 0.0 .. 2.0

    # Gaussian proximity: smooth gradient all the way in
    #   dist=0.0 → weight=1.00, dist=0.3 → 0.84, dist=0.5 → 0.61, dist=1.0 → 0.14
    proximity_weight = 2.718281828 ** (-0.5 * (dist_next / 0.5) ** 2)

    # Softer speed gate: 0.5 threshold instead of 0.2
    speed_gate = max(0.0, 1.0 - speed / 0.5)

    # Modest coefficient — smooth function means more steps get partial credit,
    # but lower peak value prevents dominance
    contact_guidance = 0.3 * contact_val * proximity_weight * speed_gate

    # ── Combine ──
    total_reward = (
        progress_reward
        + velocity_management
        + orientation_penalty
        + contact_guidance
    )

    components = {
        "progress": progress_reward,
        "velocity_management": velocity_management,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```

### 关键改动说明

| 改动 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `progress` 系数 | 3.0 | **5.0** | 加强导航主信号，抗衡 contact |
| `speed_penalty` | `-0.08 * speed_sq`（无条件） | **`-0.08 * max(0, speed - target_speed)^2`** | 远处不再惩罚必要的高速移动 |
| `descent_penalty` | `-0.8 * max(0, -vy)`（所有向下） | **`-0.6 * max(0, -vy - 0.4)`**（仅过快） | 允许安全着陆下沉，不再阻止降落 |
| `proximity` 门 | `max(0, 1 - dist/0.3)` 硬截断 | **Gaussian `exp(-0.5*(dist/0.5)^2)`** | 消除奖励悬崖，提供连续梯度 |
| `speed_gate` 阈值 | 0.2 | **0.5** | 放宽速度门，减少稀疏性 |

**预期效果：** contact_guidance 的 magnitude_share 从 70% 降至 30-40%，progress 升至 20-30%，velocity_management 从持续负值变为接近零或小幅正值（因为只在真正需要减速时才惩罚）。Checkpoint 趋势应显著平滑，不再出现 cp50% 式崩溃。
