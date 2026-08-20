# Response Record

Now I have the full picture. Let me compile the diagnosis.

---

## 诊断报告

### evidence

| 证据来源 | 数据 | 含义 |
|---|---|---|
| **iter_4 → iter_5 → iter_6 landing_bonus 演化** | 4268 → 1.27 → 0.39，active_rate 10.4% → 2.2% → 1.5% | 从 exploit 直接被压死，此后从未恢复；iter_6 的 first-contact anti-bounce gate 让 bonus 更死 |
| **iter_6 终止模式** | terminated=20/20, len=68.3, score=-124.6 | 全部 episode 在 ~68 步 crash 终止，没有任何成功着陆 |
| **iter_6 组件活跃度** | action_penalty active=3.2%, landing_bonus active=1.5% | 智能体几乎不开发动机、几乎从不触发着陆奖励 → **自由落体式 crash** |
| **iter_6 checkpoint 趋势** | score 全程在 [-115, -128] 窄带波动，零上升趋势 | 训练停滞：模型从未发现改善方向 |
| **iter_2 教训** | soft_landing_proxy=3793 (state-based, 无需接触), len=1000, terminated=0/20 | 不要求接触的 state-based 奖励导致永久盘旋 exploit |
| **iter_4 教训** | landing_bonus=4268 (transition-based, 无 anti-bounce), 98.4% magnitude_share | 无保护的 transition-based 奖励导致 bounce exploit |
| **iter_6 anti-bounce gate bug** | `prev_contacts==0.0` 阻止了第二腿接触获得 bonus（1→2 过渡时 prev_contacts≠0） | 梯度断裂：单腿触地后再触第二腿奖励为 0 |

### behavior_diagnosis

当前策略已崩溃为**自由落体 crash 模式**：智能体几乎不使用发动机（action_penalty active 仅 3.2%），随重力下落约 68 步后 crash 终止。progress_reward 贡献 ~13（约 1.1 单位距离缩减），但没有任何着陆成功信号引导它减速或调整姿态。

**根因**：landing_bonus 在两个极端之间摇摆后彻底死亡。iter_4 的 transition-based bonus 太松 → exploit (bounce farming)；iter_5/6 的太紧 → 死亡。iter_6 的 `prev_contacts==0` 反弹跳门还阻止了从单腿→双腿的自然过渡，切断了梯度。

**关键洞察**：transition-based（事件触发）bonus 天然容易被 bounce 利用，因为 bounce 产生新的 contact transition。iter_2 的 state-based 奖励避开了 bounce 问题，但不要求接触导致盘旋 exploit。正确做法是 **state-based + 必须接触**：即每一步双腿触地时给予质量奖励，这样 bounce 意味着失去接触→失去奖励，盘旋意味着无接触→无奖励。

### 干预方案 (L2)

1. **landing_bonus**：从 transition-based → **state-based per-step 着陆质量奖励**。仅当 `curr_contacts >= 2`（双腿同时触地）时给予。质量门用高斯核（距离/速度/角度）。单腿触地给 20% 探索梯度。这从根本上杜绝 bounce exploit（蹦起=失去接触=无奖励）和盘旋 exploit（无接触=无奖励）。

2. **vel_damping**：恢复 iter_4 的 punitive 速度阻尼（替代 iter_6 的 soft_approach_reward）。带垂直死区 0.8，允许受控下降。iter_4 已验证该公式有效。

3. **progress_reward / orientation_penalty / action_penalty**：保持不变，这些组件工作正常。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v7: State-based landing quality reward (per-step while both legs contact)
        replaces exploitable transition-based landing_bonus.
        Restore vel_damping (punitive) instead of soft_approach_reward.
        Single-leg contact gets 20% partial credit for exploration gradient.
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

    # ── 3. State-based landing quality reward ─────────────────────────
    # Per-step reward when legs are in contact with the pad.
    # State-based (not transition-based): no bounce exploit because
    # bouncing = losing contact = losing reward.
    # Requires contact: no hover exploit (cf. iter_2 soft_landing_proxy).
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)

    speed = (vx ** 2 + vy ** 2) ** 0.5

    # Quality gates: reward scales with how good the landing state is
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 1.0 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.5 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 20.0   # per-step weight — moderate, accumulates over settled steps
    if curr_contacts >= 2.0:
        # Both legs in contact: full landing quality reward
        landing_bonus = w_landing * quality
    elif curr_contacts >= 1.0:
        # Single-leg contact: 20% partial credit for exploration gradient
        landing_bonus = 0.2 * w_landing * quality
    else:
        landing_bonus = 0.0

    # ── 4. Directional velocity damping (punitive, near-pad) ─────────
    # Restored from iter_4: actively penalises high speed near pad,
    # with vertical deadzone to allow controlled descent.
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
