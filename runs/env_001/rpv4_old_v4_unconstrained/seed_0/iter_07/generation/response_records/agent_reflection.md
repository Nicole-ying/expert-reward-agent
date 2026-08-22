# Response Record

Now I have all the evidence I need. Let me compile the full diagnosis.

---

## 诊断报告

### Evidence

| 证据来源 | 关键数据 |
|---|---|
| **iter_4 (v4)** | score=**120.84**（当前最佳），landing_bonus=2627.5 (97.4% magnitude)，cp90% landing=5387 → cp100%=665 崩溃 |
| **iter_5 (v5)** | score=-180.10，landing/contact 均为 0 — agent 从未触地。vel_damping 扩大门控(sigma=1.5)导致全局减速惩罚 |
| **iter_6 (v6)** | score=39.15，terminated=10/20, truncated=10/20，**cp80% score=102.1 → cp90%=146.9 → cp100%=-0.7 崩溃** |
| **iter_6 checkpoint** | landing_bonus: cp80%=525.1 → cp100%=149.4（暴跌 3.5×）；action_reward: cp100%=-17.12（是 cp80% 的 2×）；contact active: cp80%=21.3% → cp100%=5.2% |
| **iter_6 final** | landing_bonus=183.6 (78.6% magnitude, active_rate=1.9%)，说明大多数 episode 几乎没有成功着陆 |
| **v4 vs v6 共同模式** | 两轮都在 cp80-90% 达到峰值后**崩溃到接近零或负** — 这是经典的 reward exploitation → policy collapse 模式 |

### Behavior Diagnosis

**核心问题：landing_bonus 的二元悬崖结构导致训练不稳定 → 策略崩溃。**

v4（per-step 50）和 v6（transition 200 + sustain 5 + near-landing 2）虽然形式不同，但共享同一个根本缺陷：**landing_bonus 的量级远超其他所有组件之和**（v4: 97.4%，v6: 78.6%），使得整个 reward landscape 被一个稀疏/半稀疏的巨大信号主导。

训练动力学：
1. agent 随机探索，偶尔触发 landing 条件 → 获得巨大奖励脉冲
2. 策略快速向该脉冲方向偏移 → 过度拟合到触发 landing 的具体状态序列
3. 随着持续训练和探索噪声，脆弱的策略崩溃 → agent "忘记"如何着陆
4. cp100% 时 action_reward=-17.12（远高于 cp80% 的 -8.78），说明崩溃后的 agent 在疯狂燃烧引擎做无用功

v6 的 transition detection 试图用「首次触发给 200，后续仅给 5」来抑制 exploit，但它没有解决根本问题：**200 本身仍然是巨大的稀疏脉冲**，在 reward space 中制造了一个悬崖。

另外，v6 还有 **50% episode 被 truncate**（10/20），说明 agent 即使着陆也无法可靠触发环境的 settled 终止条件 —— 着陆后还在消耗动作维持姿态，最终超时。

### 干预策略：连续着陆质量函数

**核心思路：将二元 landing_bonus 替换为连续、平滑的着陆质量分数。**

- 用各因子的乘积构建 0~1 的连续质量分：`dist_quality × angle_quality × speed_quality × contact_quality`
- 最大 per-step 奖励控制在 **12.0**（完美着陆时），远低于 v4 的 50 和 v6 的 200
- 连续函数提供处处可导的梯度，没有悬崖 —— agent 可以通过逐步改善每个因子来提升奖励
- 仅在至少一只脚触地时激活（`any_contact` gate），避免悬停也能得分
- 保留小量 contact_reward（0.5/foot）作为触地探索引导
- progress_reward 从 1.5 微调至 2.0，强化导航梯度以弥补 landing 信号的减弱
- 其余组件保持 v4 验证过的尺度

预期效果：landing_bonus magnitude_share 从 78-97% 降至 40-60%，训练曲线不再出现 cp90%→cp100% 崩溃。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7: Continuous landing quality — eliminate binary cliffs that cause policy collapse.

    Root cause of v4/v6 crashes: landing_bonus was a sparse/semi-sparse spike (50/step
    or 200-transition) that dominated the reward landscape. Agent overfit to triggering
    it, then the brittle policy collapsed under continued exploration.

    Fix: Replace binary landing check with a continuous 0–1 quality score.
    - Product of 4 sub-scores: dist, angle, speed, contact — all must be good for high reward
    - Max 12.0/step (vs 50 in v4, 200+5 in v6) — balanced against other components
    - Only activates when at least one foot touches ground (no hovering credit)
    - Contact reward (0.5/foot) kept for ground-discovery gradient
    - Progress boosted to 2.0 to compensate for reduced landing signal
    - All other components at v4-proven scales
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    px = obs[0]
    py = obs[1]

    # ── Derived quantities ──
    dist = (x ** 2 + y ** 2) ** 0.5
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # ── 1. Progress: delta-distance (slightly boosted for navigation gradient) ──
    progress_reward = 2.0 * (prev_dist - dist)

    # ── 2. Orientation penalty (v4 scale: proven stable) ──
    orientation_penalty = -0.1 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 3. Velocity damping near pad (v4 scale: sigma=0.5, coeff=-0.5) ──
    proximity_gate = 2.718281828 ** (- (dist ** 2) / 0.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * proximity_gate * excess_speed

    # ── 4. Contact reward: modest ground-discovery incentive ──
    contact_reward = 0.5 * (left_contact + right_contact)

    # ── 5. Continuous landing quality (replaces binary landing_bonus) ──
    any_contact = (left_contact > 0.5 or right_contact > 0.5)

    if any_contact:
        # Each sub-score: 0 (worst) to 1 (perfect), linear decay outside target zone
        dist_quality = max(0.0, 1.0 - dist / 0.5)          # 1 at pad center, 0 at dist>=0.5
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.4)   # 1 when upright, 0 at |angle|>=0.4
        speed_quality = max(0.0, 1.0 - speed / 0.5)        # 1 when still, 0 at speed>=0.5
        contact_quality = (left_contact + right_contact) / 2.0  # 0, 0.5, or 1.0

        # Product: ALL factors matter — partial credit for partial progress
        landing_quality = dist_quality * angle_quality * speed_quality * contact_quality
        landing_bonus = 12.0 * landing_quality
    else:
        landing_bonus = 0.0

    # ── 6. Action efficiency (v4 scale) ──
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
