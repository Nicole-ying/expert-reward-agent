# Response Record

现在我有了完整的图景。让我输出诊断和代码。

---

## 诊断报告

### Evidence

| 证据来源 | 关键数据 |
|---|---|
| **landing_bonus 历史** | iter_4: 97.4% mag (2627 ep_sum, 坐地 exploit)；iter_7: 81.0% (262)；iter_8: **94.7% (922 ep_sum)** — 一次性的修复失败了 |
| **iter_8 checkpoint 轨迹** | cp60% landing_bonus=**0**（崩溃）→ cp90%=**926.8**（弹跳 exploit）→ cp100%=**1188.5**（加剧） |
| **iter_8 score 轨迹** | cp60%=**-481.7**（灾难性） → cp90%=-22.5（恢复但带 exploit）→ final=-53.83 |
| **iter_8 最终评估** | terminated=14/20 (70% 着陆)，但弹跳 exploit 导致 landing_bonus=921.5 / 每集 (~12 次着陆) |
| **所有其他组件** | progress=1.24, orientation=-1.22, vel_damping=-1.40, action=-14.11 — 在 922 的 landing_bonus 面前全部微不足道 |
| **最佳迭代** | iter_4: score=120.84，但 landing_bonus=97.4% 占比且 cp90% 同样崩溃 |

### Behavior Diagnosis

**v8 的"一次性"着陆奖励被弹跳 exploit 攻破。** 机制：`just_touched_down = was_airborne and touching_now` 每次飞行器弹跳离开地面再落下都会重新触发。智能体学会了反复弹跳以重复获得奖励（~12 次着陆/集），而非稳定停在垫子上。这解释了为何 active_rate 仅 3.7% 但 ep_sum 高达 922：每次触发都获得接近最大值（~80），只需触发 ~12 次。

**更深层的问题：** 所有包含离散着陆奖励的迭代（v4/v6/v7/v8）——无论按步发放还是一次性发放——最终都被 exploit。v4 的智能体坐在垫子上（18% active_rate × 50/步 = 2627 ep_sum），v8 的智能体弹跳（3.7% × ~80/次 = 922 ep_sum）。**离散事件奖励本质上可以被 agent 利用——它总能找到一种重复触发的方式。**

**根本解决方案：彻底移除离散着陆奖励。** 用连续的、不可利用的替代方案：一个始终激活的高斯邻近度奖励，在着陆垫处达到峰值。智能体不能通过弹跳来利用它——弹跳离开垫子会降低奖励。最优策略就是停在 (0,0) 处，这正是我们期望的行为。

### Intervention

**框架级重写：离散着陆奖励 → 连续邻近度奖励。**

- **移除 `landing_bonus`**（一次性着陆检测 + 大额奖励）——这是 exploit 的根源
- **新增 `proximity_reward`**：始终激活的高斯函数 `0.5 * exp(-dist²/0.8)`，在垫子处达到最大值 0.5/步
- **保留并微调**：progress（导航信号）、orientation（姿态）、vel_damping（软着陆）、contact（触地可发现性）、action（效率）
- **缩放**：proximity 和 contact 设计为共同提供 ~200-300 ep_sum（共同主导但不过量），所有其他信号在几十的量级，占比更均衡
- **为何有效**：邻近度 + 接触的组合在垫子上创造约 1.0/步的奖励，而在远处仅约 0.05/步——形成了 20:1 的梯度，且没有可被利用的离散事件

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v9: Continuous proximity replaces discrete landing bonus.

    ROOT CAUSE OF v4-v8 FAILURES:
    Every iteration with a discrete "landing bonus" (per-step or one-time) was exploited:
    - v4: agent sat on pad collecting 50/step → landing_bonus=2627, 97.4% magnitude
    - v8: agent bounced to re-trigger one-time bonus → landing_bonus=922, 94.7% magnitude
    Discrete events are inherently exploitable — the agent always finds a way to re-trigger them.

    v9 APPROACH:
    - NO discrete landing bonus. NO touchdown detection. NO "just_touched_down" triggers.
    - proximity_reward: continuous Gaussian centered on pad (0,0). Always active, smooth
      gradient, max 0.5/step at center. Cannot be exploited: bouncing AWAY reduces reward.
    - On-pad: proximity + contact ≈ 1.0/step. Far away: ≈ 0.05/step. 20:1 natural gradient.
    - All components are continuous → no exploit cliff, no collapse-recovery cycle.
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

    # ── 1. Progress: delta-distance (navigation signal, proven in all iterations) ──
    progress_reward = 2.0 * (prev_dist - dist)

    # ── 2. Proximity: continuous Gaussian pull toward pad center ──
    # Replaces the exploitable discrete landing bonus.
    # max 0.5 at (0,0), decays smoothly: 0.37 at dist=0.5, 0.14 at dist=1.0, ~0 at dist=2.0
    # Always active → agent always feels gradient toward pad. Bouncing reduces reward.
    proximity_reward = 0.5 * (2.718281828 ** (-dist ** 2 / 0.8))

    # ── 3. Orientation penalty: stable, always active, proven scales ──
    orientation_penalty = -0.15 * (angle ** 2) - 0.06 * (angvel ** 2)

    # ── 4. Velocity damping near pad: prevent hard landings ──
    near_pad = 2.718281828 ** (-dist ** 2 / 0.5)
    safe_speed = 0.25
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * near_pad * excess_speed

    # ── 5. Contact: reward ground interaction for discoverability ──
    contact_reward = 0.3 * (left_contact + right_contact)

    # ── 6. Action efficiency: gentle penalty, glide is free ──
    if action == 0:
        action_reward = 0.0
    elif action == 2:          # main engine — highest cost
        action_reward = -0.04
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.02

    # ── Aggregate ──
    total_reward = (
        progress_reward
        + proximity_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + action_reward
    )

    components = {
        "progress_reward": progress_reward,
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "action_reward": action_reward,
    }
    return float(total_reward), components
```
