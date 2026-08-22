## 诊断报告

### evidence

| 证据来源 | 数据 | 含义 |
|---|---|---|
| **iter_4 → iter_5/6/7 轨迹** | score: 120 → -121 → -125 → -124 | 在 iter_4 发现可行路径后，iter_5-7 试图修复 exploit 却把唯一有效梯度杀死了 |
| **iter_4 checkpoint 趋势** | landing_bonus 从 14→4264，score 从 -107→+112（cp80% 峰值 133） | iter_4 的 transition-based landing_bonus 被 bounce exploit 放大 300x；但至少证明了 agent **有能力学会接近并接触着陆垫** |
| **iter_7 checkpoint 趋势** | score 全程 -120 附近，零波动 | 训练完全停滞：agent 从未发现任何改善方向 |
| **iter_7 终止模式** | terminated=20/20, len=68.3 | 所有 episode 快速坠毁，agent 从未到达着陆垫 |
| **vel_damping 历史** | iter_1:0 → iter_2:-26 → iter_3:-20 → iter_4:-7.4 → iter_5-7:-7.1 | vel_damping 始终是纯惩罚信号；在 iter_7 中 magnitude_share=24.6%，是第二大信号 |
| **iter_7 landing_bonus** | ep_sum=6.54, active_rate=3.3% | 状态型设计本身正确（防 bounce），但权重太低不足以被发现 |
| **iter_6 soft_approach** | w=0.4, ep_sum≈0.07 | 正面试过 soft_approach 但权重太弱，被其他信号淹没 |

### behavior_diagnosis

当前策略的行为模式：**快速坠毁**。agent 学会了「不要靠近着陆垫」——因为 iter_7 的 `vel_damping` 在靠近着陆垫时惩罚速度（gate × |vx| + max(0,|vy|-0.8)），形成了「接近=受罚」的条件反射。progress_reward (+13.4/ep) 是唯一正向信号，但不足以对抗 vel_damping (-7.1) + orientation (-1.4) 的惩罚合力。agent 的最优策略是：飞几秒拿一点 progress，然后在 vel_damping 生效前坠毁。这解释了 68 步的短 episode 和 -124 的分数。

**根本问题**：iter_5-7 连续三轮都在「惩罚靠近着陆垫的速度」，但没有提供任何**正向梯度**引导 agent 减速靠近。vel_damping 告诉 agent「靠近时速度快=不好」，但不说「靠近时速度慢=好」。这导致 agent 选择「干脆不靠近」。

### 修改方案（L2 形态替换 + L1 权重配合，同一语义方向）

**语义方向**：将「惩罚靠近时的速度」(vel_damping) 替换为「奖励靠近时的低速」(soft_approach_reward)，同时强化着陆奖励使触地明确优于悬停。

- **删除 vel_damping**：惩罚性信号已被证明抑制靠近行为
- **新增 soft_approach_reward**（w=2.0）：正向高斯奖励，在靠近着陆垫且速度低时给予正反馈，形成「接近→减速→接触」的连续梯度漏斗
- **提升 landing_bonus 权重**（20→40）：确保接触后的奖励远超悬停，消除 hover exploit 风险

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v8: Replace punitive vel_damping with positive soft_approach_reward —
        a gradient that pulls the agent toward the pad AND rewards deceleration.
        Boost state-based landing_bonus to make touchdown clearly better than hovering.
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

    # ── 3. Soft approach reward (POSITIVE gradient near pad + low speed) ─
    # Replaces punitive vel_damping: agent earns reward for being close
    # AND slow, creating a funnel: approach → decelerate → contact.
    # Contact-independent so it guides behaviour BEFORE first touchdown.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    w_soft = 2.0
    sigma_dist = 1.5                                    # landing-zone radius
    sigma_speed = 0.5                                   # acceptable approach speed
    soft_approach_reward = (
        w_soft *
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * sigma_dist ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * sigma_speed ** 2))
    )

    # ── 4. State-based landing quality reward ─────────────────────────
    # Per-step reward when legs are in contact with the pad.
    # State-based (not transition-based): no bounce exploit because
    # bouncing = losing contact = losing reward.
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)

    # Quality gates: reward scales with how good the landing state is
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 1.0 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.5 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 40.0                                    # boosted from 20 → strong touchdown incentive
    if curr_contacts >= 2.0:
        # Both legs in contact: full landing quality reward
        landing_bonus = w_landing * quality
    elif curr_contacts >= 1.0:
        # Single-leg contact: 20% partial credit for exploration gradient
        landing_bonus = 0.2 * w_landing * quality
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