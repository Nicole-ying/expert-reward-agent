# Response Record

## 诊断报告

### evidence

| 证据来源 | 数据 | 含义 |
|---|---|---|
| **iter_8 分数与终止** | score=-37.7, len=1000, terminated=0/20, truncated=20/20 | 全部 episode 跑满 1000 步被截断，无一次着陆。智能体在盘旋 |
| **iter_8 soft_approach_reward** | ep_sum=1692, magnitude_share=**91.9%** | 该组件完全支配 reward，成为唯一有效信号 |
| **soft_approach_reward 历史跳跃** | iter_6: 4.7 → iter_8: 1692 (**356x**) | 指数级增长 → 被 exploit |
| **iter_8 landing_bonus** | ep_sum=0.0, active_rate=**0.0%**，全程 0 | 完全死亡，智能体从未触发着陆接触 |
| **iter_8 checkpoint 趋势** | soft_approach 从 39 单调增长到 1693，score 从 -1774 升到 -36 | soft_approach 提供了接近+减速的梯度，但到了 pad 附近就停住盘旋 |
| **iter_2 历史对照** | iter_2 用 soft_landing_proxy（连续状态值奖励），score=-100, len=1000, 0/20 terminated | **完全相同的悬停 exploit 模式**：状态值奖励 → 盘旋吃分不落地 |
| **iter_4 (唯一正分)** | score=**120.17**, len=467, terminated=16/20 | transition-based landing_bonus + progress + vel_damping 框架的可行性被证明 |
| **iter_4 landing_bonus** | ep_sum=4268, magnitude_share=98.4%, quality gates sigma_dist=1.5/sigma_speed=0.5 | exploit 形式是 **bounce**（反复弹跳重复触发 transition bonus），而非悬停 |
| **iter_4 checkpoint** | landing_bonus cp40% 开始暴增，cp40%=1330, cp80%=3304, score cp80%=**132.9** | 智能体确实学会了接近→减速→着陆，只是 quality gates 太松导致弹跳可获利 |
| **iter_5 (紧 gates)** | sigma_dist=0.8, sigma_speed=0.3, w=30 → landing_bonus 死亡(ep_sum=1.3, active=2.2%), score=-121 | quality gates 收紧过度导致着陆信号消失 |
| **iter_6/7/8 landing_bonus** | active_rate 从 1.5%→3.3%→0%，持续死亡 | 无论 transition 还是 state-based，只要 soft_approach 存在就永远发现不了着陆 |

### behavior_diagnosis

**当前策略行为：悬停 exploit（iter_2 的翻版）**

智能体学会了：接近目标着陆垫 → 减速至近乎零 → 在 pad 正上方悬停 1000 步。soft_approach_reward 的高斯函数在 dist≈0, speed≈0 处达到最大值 2.0/step，这意味着悬停 1000 步可获得约 2000 的总分，远超任何可能的着陆奖励（即使 landing_bonus 工作，一次着陆最多几十点）。**这不是"接近目标的进步"，而是陷入完全错误的局部最优**——智能体永远不落地。

**根因**：任何"在目标位置附近停留就能持续获得正奖励"的 state-based reward，都不可避免地被 exploit 为悬停。这在 iter_2 (soft_landing_proxy) 和 iter_8 (soft_approach_reward) 中两次被证实。

### 修复方向

回到 **iter_4 的被证明可行的框架**（transition-based landing_bonus + progress_reward + vel_damping），做三件事：

1. **完全移除 soft_approach_reward** — 它不可避免地导致悬停 exploit
2. **landing_bonus 的 quality gates 取 iter_4 和 iter_5 之间的中间值**：sigma_dist=1.0（iter_4=1.5 太松/iter_5=0.8 太紧），sigma_speed=0.5（iter_4=0.5 刚好/iter_5=0.3 太紧）——让 rough landing 也能获得小奖励以保证可发现性，但 bounce 因速度高而几乎无奖励
3. **添加 first-contact gate**（prev_contacts==0 时才能触发）：每个 episode 只奖励第一次接触，彻底消灭 bounce exploit。iter_4 证明过渡型 landing_bonus 可被发现；现在用 first-contact gate 消除 exploit 而保留发现性
4. **提升 vel_damping 权重至 0.4**（从 iter_4 的 0.2）：让减速惩罚成为有意义的梯度，替代 soft_approach 的减速引导作用。配合 landing_bonus 的 quality 函数（奖励低速接触），形成双重减速压力

**修改层级**：L2 — 移除一个组件（soft_approach_reward），恢复 iter_4 框架并调整 landing_bonus 的数学形态（first-contact gate + 中间 quality gates），调整 vel_damping 权重。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v9: Back to iter_4's proven framework (transition landing_bonus + progress + vel_damping).
        Remove soft_approach_reward entirely — it caused hover exploit in both iter_2 and iter_8.
        landing_bonus: first-contact gate (prev_contacts==0) kills bounce exploit;
        quality gates at intermediate tightness (sigma_dist=1.0, sigma_speed=0.5) —
        discoverable but bounce-unprofitable.
        Boost vel_damping (w=0.4) to provide meaningful deceleration gradient.
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

    # ── 3. Landing contact bonus (transition-based, first-contact-only) ─
    # Count leg contacts before and after this step
    prev_contacts = (1.0 if obs[6] > 0.5 else 0.0) + (1.0 if obs[7] > 0.5 else 0.0)
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)
    new_contacts = max(0.0, curr_contacts - prev_contacts)   # 0, 1, or 2 newly touching

    # Quality gates: between iter_4 (1.5/0.5 — loose, bounce-exploitable)
    # and iter_5 (0.8/0.3 — tight, undiscoverable).
    # These values: rough landing earns ~8-15, clean landing ~50-60, bounce ~0-1.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 1.0 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.5 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 60.0
    # First-contact-only gate: bonus fires ONLY when no leg was touching before.
    # This kills the iter_4 bounce exploit — once contact is made, no more transition bonuses.
    # prev_contacts == 0 ensures the agent can't bounce-and-re-trigger.
    if prev_contacts == 0.0 and new_contacts >= 2.0:
        # Both legs touch simultaneously for the first time: full bonus
        landing_bonus = w_landing * quality
    elif prev_contacts == 0.0 and new_contacts >= 1.0:
        # Single-leg first contact: 30% partial credit for exploration gradient
        landing_bonus = 0.3 * w_landing * quality
    else:
        landing_bonus = 0.0

    # ── 4. Directional velocity damping (punitive, near-pad) ─────────
    # Boosted from w=0.2 (iter_4, negligible) to w=0.4 — provides meaningful
    # deceleration gradient as soft_approach replacement.
    # Vertical deadzone (0.6) allows controlled descent without penalty.
    dist_thresh = 3.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 far away, 1 at origin

    horizontal_penalty = abs(vx)                         # full penalty on lateral drift
    vertical_penalty = max(0.0, abs(vy) - 0.6)           # deadzone: allow moderate descent

    w_vel_damp = 0.4                                     # up from 0.2 → meaningful decel gradient
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
