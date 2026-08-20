## 诊断报告

### evidence

| 证据来源 | 数据 | 含义 |
|---|---|---|
| **iter_4 → iter_5 → iter_9 轨迹** | score: 120 → -121 → ... → -117 | iter_4 是唯一成功的迭代；iter_5-9 全部失败 |
| **landing_bonus 演化** | iter_4: 4268 (act=10.4%) → iter_5: 1.27 → iter_9: 0.96 (act=1.5%) | 从 iter_5 起 landing_bonus 被杀死，此后从未恢复 |
| **iter_9 landing_bonus** | ep_sum=0.958, active_rate=1.5%, magnitude_share=2.8% | 几乎不触发，即使触发也微不足道 |
| **iter_9 vel_damping** | ep_sum=-17.09, magnitude_share=50.1%, w=0.4 | 是 iter_4 (w=0.2, ep=-7.4) 的 2.3 倍，成为主导惩罚信号 |
| **iter_9 progress_reward** | ep_sum=13.37, magnitude_share=40.6% | 纯几何量，训练全程不变（cp10%-cp100% 均为 ~13.37） |
| **iter_9 终止模式** | terminated=20/20, len=69.75, 18/20 early_terminal | 全部 episode 快速坠毁，无着陆行为 |
| **iter_4 checkpoint** | score 从 cp10%=-107 增长到 cp80%=133 | landing_bonus 从 14→3304，证明了 transition-based 结构的可学习性 |
| **iter_9 checkpoint** | score 全程 -114~-120，零改善 | 训练完全停滞：agent 找不到任何正向梯度 |

### behavior_diagnosis

**当前策略行为：快速坠毁。** 18/20 episode 在 150 步内以 score<-50 终止。agent 没有尝试着陆——它直接坠向地面。

**根因分析**：iter_9 试图"回到 iter_4 框架"但加了 `prev_contacts==0` 的 first-contact gate。这个 gate 导致 landing_bonus 每 episode 最多触发一次，配合较紧的 quality gates（sigma_dist=1.0），实际触发时 quality 也极低（ep_sum=0.96）。与此同时 vel_damping 被加倍到 w=0.4（ep_sum=-17.1），与 progress_reward（+13.4）相抵后净 reward 为负。agent 的唯一可学习正向信号（landing_bonus）被 gate 杀死了，因此它只学会了一件事：快速结束 episode 以减少累积惩罚。

**核心矛盾**：iter_4 的 transition-based landing_bonus（无 gate）是可学习的，但会导致 bounce exploit（magnitude_share=98.4%）。iter_5-9 的所有"修复"都通过杀死信号来防止 exploit，结果 agent 退化为 crash。**我们需要中间地带：保留 iter_4 的可学习结构，但降低权重以防止 domination。**

### 干预方案（L2）

**单一语义修改**：将 landing_bonus 从"first-contact-gated"恢复为 iter_4 的"transition-based"结构，同时降低权重（80→50）防止 magnitude  domination，并配套降低 vel_damping（0.4→0.2）恢复 iter_4 的平衡。这本质上是"回到 iter_4 + 适度降权"。

具体变更：
1. **landing_bonus**：移除 `prev_contacts==0` gate → `w_landing * new_contacts * quality`；sigma_dist 1.0→1.5（iter_4 值）；w_landing 60→50
2. **vel_damping**：w 0.4→0.2（iter_4 值），vertical deadzone 0.6→0.8（iter_4 值）

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v10: Revert to iter_4's transition-based landing_bonus (no first-contact gate)
         with moderate weight (50 vs iter_4's 80) to preserve learnability while
         reducing bounce-exploit domination. Revert vel_damping to iter_4 levels
         (w=0.2, deadzone=0.8) to stop crushing the agent.
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

    # ── 3. Landing contact bonus (transition-based, iter_4 structure) ─
    # Count leg contacts before and after this step
    prev_contacts = (1.0 if obs[6] > 0.5 else 0.0) + (1.0 if obs[7] > 0.5 else 0.0)
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)
    new_contacts = max(0.0, curr_contacts - prev_contacts)   # 0, 1, or 2 newly touching

    # Quality gates at iter_4 parameters: loose enough to be discoverable,
    # but the bonus magnitude is moderated by w_landing=50 (vs iter_4's 80).
    speed = (vx ** 2 + vy ** 2) ** 0.5
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 1.5 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.5 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 50.0                                   # moderate: between iter_4 (80) and iter_5 (30)
    landing_bonus = w_landing * new_contacts * quality  # no first-contact gate → discoverable

    # ── 4. Directional velocity damping (iter_4 parameters) ──────────
    dist_thresh = 3.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 far away, 1 at origin

    horizontal_penalty = abs(vx)                         # full penalty on lateral drift
    vertical_penalty = max(0.0, abs(vy) - 0.8)           # deadzone: allow moderate descent

    w_vel_damp = 0.2                                     # iter_4 value (not 0.4)
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