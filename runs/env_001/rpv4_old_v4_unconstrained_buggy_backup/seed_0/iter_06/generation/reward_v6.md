## 诊断报告

### evidence

| 证据来源 | 数据 | 含义 |
|---|---|---|
| iter_4→iter_5 score 骤降 | 120.17 → -121.30，delta=-241 | iter_5 的修改过度抑制了 landing signal，策略崩溃 |
| landing_bonus 演化 | iter_3:12.4 → iter_4:4268(exploit) → iter_5:1.27(死) | 从 exploit 直接掉到近乎零，中间值从未被探索 |
| iter_5 checkpoint 趋势 | score 全程 -120 左右，零波动 | **训练停滞**：模型从未发现任何改善方向 |
| iter_5 终止模式 | terminated=20/20，len=68.3 | 所有 episode 都是早期坠毁，没有一次接近成功着陆 |
| vel_damping vs progress_reward | vel=-7.11 (30.2%) vs progress=+13.39 (58.9%) | vel_damping 吃掉了 progress 的一半以上，净激励很弱 |
| landing_bonus active_rate | 2.2%，ep_sum=1.27 | 即使触发也几乎无奖励（质量门太紧），等于没有着陆信号 |
| iter_4 landing_bonus 趋势 | cp10%=14 → cp100%=3982，持续增长 | 信号强度本身驱动了学习，问题是 exploit 而非结构 |
| iter_4 score 趋势 | cp10%=-107 → cp80%=133 → cp100%=112 | 训练中段曾达到 133，说明可学习，但 exploit 导致后期退化 |

### behavior_diagnosis

当前策略行为模式：**受 progress_reward 驱动冲向目标垫，但完全不学习减速**。因为：
- `progress_reward` (+13.4) 是唯一显著的正向信号
- `vel_damping` (-7.1) 是纯惩罚，agent 学会"忍受"而非"避免"它——因为减速会导致 progress 变慢，净收益为负
- `landing_bonus` (1.27, active_rate 2.2%) 实际已死——质量门过紧（sigma_speed=0.3）+ 权重过低(30)，即使软着陆也几乎无奖励
- 最终结果：68 步高速撞地，从未体验过成功着陆

**核心矛盾**：iter_4 证明 agent 能学到着陆行为（score=120），但 landing_bonus 被 exploit（弹跳重复领取）。iter_5 为防止 exploit 把 landing_bonus 杀死了。需要一个**中间方案**：适中的 landing 信号 + 更强的减速正面引导。

### 干预选择：L2（单语义方向）

**修改内容**：将 `vel_damping`（纯惩罚→惩罚减速）替换为 `soft_approach_reward`（正向奖励→奖励低速），同时调整 `landing_bonus` 到 iter_4 和 iter_5 之间的合理区间。

**理由**：纯惩罚的 vel_damping 在 5 轮迭代中从未有效引导减速（agent 宁愿接受惩罚也不减速，因为减速会牺牲 progress）。将「靠近目标时低速」从惩罚变为奖励，创造正向学习梯度。同时将 landing_bonus 质量和权重放宽到适中水平，让成功着陆重新可见。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v6: Replace punitive vel_damping with positive soft_approach_reward
        (reward low speed near pad instead of penalising speed);
        restore landing_bonus to moderate weight + moderate quality gates
        (between iter_4's loose/exploitable and iter_5's tight/dead);
        differentiate full landing vs partial contact, anti-bounce via first-contact gate.
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

    # ── 3. Soft approach reward (POSITIVE reward for low speed near pad) ─
    # Replaces punitive vel_damping: agent earns reward for decelerating,
    # creating a positive gradient instead of a cost to be "tolerated".
    dist_thresh = 3.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 far away, 1 at origin
    speed = (vx ** 2 + vy ** 2) ** 0.5
    w_soft = 0.4
    sigma_speed = 0.6                                    # moderate: rewards meaningful slowing
    soft_approach_reward = (
        w_soft * gate *
        2.718281828 ** (- (speed ** 2) / (2.0 * sigma_speed ** 2))
    )

    # ── 4. Landing contact bonus (transition-based, anti-bounce) ─────
    prev_contacts = (1.0 if obs[6] > 0.5 else 0.0) + (1.0 if obs[7] > 0.5 else 0.0)
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)
    new_contacts = max(0.0, curr_contacts - prev_contacts)

    # Moderate quality gates: between iter_4 (1.5/0.5 → loose, exploitable)
    # and iter_5 (0.8/0.3 → tight, dead).  Graded enough to guide learning
    # but suppress high-speed bounce re-trigger.
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 1.0 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.4 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 45.0                                     # between 30 (dead) and 80 (exploit)
    # Anti-bounce gate: only reward FIRST contact transition (prev_contacts == 0).
    # Once any leg has touched, subsequent re-contacts earn no bonus.
    # This blocks the iter_4 bounce-exploit while preserving the initial touchdown signal.
    if prev_contacts == 0.0 and new_contacts >= 2.0:
        landing_bonus = w_landing * quality
    elif prev_contacts == 0.0 and new_contacts >= 1.0:
        landing_bonus = 0.3 * w_landing * quality        # partial credit for first single-leg touch
    else:
        landing_bonus = 0.0

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
        soft_approach_reward +
        landing_bonus +
        action_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "soft_approach_reward": soft_approach_reward,
        "landing_bonus": landing_bonus,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```