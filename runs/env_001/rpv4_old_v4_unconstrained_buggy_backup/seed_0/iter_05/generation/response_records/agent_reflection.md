# Response Record

现在我已经掌握了所有必要的数据。让我进行全面诊断。

---

## 诊断报告

### Evidence

| 证据来源 | 数据 | 含义 |
|---|---|---|
| landing_bonus magnitude_share | **98.4%**，ep_sum=4268 | 训练奖励完全被 landing_bonus 支配，其他组件沦为噪声 |
| landing_bonus 历史跳跃 | iter_3: 12.4 → iter_4: 4268，**343x 暴增** | 组件被 exploit，不是正常的信号放大 |
| episode length | **467 steps**（iter_3 仅 69） | 智能体没有快速稳定着陆，而是在反复操作 |
| terminated/truncated | 16/20 terminated，**4/20 truncated**（跑满 1000 步） | 部分 episode 永远不终止，符合"弹跳 exploit"模式 |
| landing_bonus active_rate | 10.4%（~48 次接触事件/episode） | 远超正常单次着陆应有的频率（1-2 次） |
| score checkpoint 趋势 | cp70%=64 → cp80%=**133** → cp90%=**52** → cp100%=112 | 晚期剧烈波动，典型的 proxy 与真实任务目标背离 |
| landing_bonus checkpoint | cp30%=40 → cp40%=**1330**（33x 跳变） | cp40% 时智能体发现了接触 exploit，此后一路飙升 |
| progress_reward magnitude_share | 仅 **0.3%** | 被 landing_bonus 完全淹没，失去引导作用 |
| vel_damping magnitude_share | 仅 **0.2%** | 同样被淹没 |

### Behavior Diagnosis

智能体当前学到的行为模式：**"弹跳 exploit"**。方向性速度阻尼和进度奖励成功教会了智能体接近着陆垫（这是好的部分），但过大的 `landing_bonus` 劫持了后续行为——智能体发现反复触发接触事件（接触→弹起→再接触→再弹起）可以获取海量奖励，而不是真正稳定着陆。

具体表现：
- 平均每 episode **48 次新接触事件**，远超正常着陆的 1-2 次
- 467 步的 episode 长度说明智能体在反复弹跳而非快速稳定
- 4/20 episode 跑满 1000 步——这些是永远无法稳定的弹跳循环
- 训练晚期 score 剧烈波动（52↔133），因为 agent 在"最大化 proxy（弹跳）"和"真正着陆"之间摇摆

### 干预方向

**L2 修改**：重构 `landing_bonus`，三个改动服务于同一语义——**抑制弹跳 exploit，保留着陆引导信号**：

1. **降低权重**：`w_landing` 80 → 30，大幅削减其在总奖励中的占比
2. **收紧质量门控**：距离 σ 1.5→0.8，速度 σ 0.5→0.3。只有非常靠近中心且速度极低时才能获得高质量奖励，弹跳（速度较高）自然被抑制
3. **区分双腿/单腿接触**：`new_contacts >= 2`（双腿同时）给全额奖励，`new_contacts == 1`（单腿）仅给 25%。弹跳往往是一腿先着地再另一腿，这使得 exploit 收益大幅降低

---

## 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v5: Reduce landing_bonus weight (80→30), tighten quality gates
        (dist sigma 1.5→0.8, speed sigma 0.5→0.3),
        differentiate full landing (both legs) vs partial (one leg, 25%).
        Goal: suppress bounce exploit while preserving touchdown signal.
    """
    # ── Extract signals ──────────────────────────────────────────────
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # ── 1. Progress reward (potential-based: bounded total, no hover exploit) ─
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # >0 when approaching
    w_progress = 12.0
    progress_reward = w_progress * progress

    # ── 2. Orientation penalty ───────────────────────────────────────
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ── 3. Landing contact bonus (transition-based, anti-bounce) ─────
    prev_contacts = (1.0 if obs[6] > 0.5 else 0.0) + (1.0 if obs[7] > 0.5 else 0.0)
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)
    new_contacts = max(0.0, curr_contacts - prev_contacts)   # 0, 1, or 2 newly touching

    # Landing quality: tighter gates to suppress bounce exploitation
    speed = (vx ** 2 + vy ** 2) ** 0.5
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 0.8 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.3 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 30.0                                    # reduced from 80 → curb domination
    if new_contacts >= 2.0:
        # Both legs touched simultaneously: proper landing → full bonus
        landing_bonus = w_landing * quality
    elif new_contacts >= 1.0:
        # Single-leg contact: partial credit (25%) for exploration, not exploitation
        landing_bonus = 0.25 * w_landing * quality
    else:
        landing_bonus = 0.0

    # ── 4. Directional velocity damping ──────────────────────────────
    dist_thresh = 3.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 far away, 1 at origin

    horizontal_penalty = abs(vx)                         # full penalty on lateral drift
    vertical_penalty = max(0.0, abs(vy) - 0.8)           # deadzone: allow moderate descent

    w_vel_damp = 0.2
    vel_damping = -w_vel_damp * gate * (horizontal_penalty + vertical_penalty)

    # ── 5. Action efficiency ─────────────────────────────────────────
    if action == 0:                                      # no engine
        action_penalty = 0.0
    elif action == 1 or action == 3:                     # orientation engines
        action_penalty = -0.05
    else:                                                # action == 2, main engine
        action_penalty = -0.2

    # ── Assemble ─────────────────────────────────────────────────────
    total_reward = (
        progress_reward +
        orientation_penalty +
        landing_bonus +
        vel_damping +
        action_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus,
        "vel_damping": vel_damping,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```
