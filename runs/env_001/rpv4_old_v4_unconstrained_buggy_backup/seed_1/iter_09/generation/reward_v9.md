---

## 诊断报告

### Evidence

| 来源 | 关键数据 |
|---|---|
| **跨迭代演化** | v1(-3)→v2(67)→v3(128)→**v4(135, BEST, 19/20)**→v5(-113)→v6(-25)→v7(-116)→v8(14) |
| **v4 成功因子** | proximity=40, soft_landing_bonus active_rate=23.5%, ep_sum=161.4, 76.2% mag — 着陆奖励主导但健康 |
| **v8 失败因子** | proximity=30, soft_landing_bonus active_rate=**3.4%**, ep_sum=39.1, 41.9% mag — 着陆奖励太稀疏 |
| **soft_landing_bonus 演化** | v1(0.3%)→v2(3.9%)→v3(5.8%)→**v4(23.5%)**→v5(1.5%)→v6(0%)→v7(1.8%)→v8(3.4%) |
| **proximity_progress 演化** | v1-v3(~33@coeff=30)→**v4(44@coeff=40)**→v5(56@50)→v8(25@30) |
| **v8 checkpoint 趋势** | soft_landing_bonus: cp50%=40→cp60%=**80.7**→cp70%=15→cp100%=27.7；score: cp60%=**87**→cp100%=-20.4 |
| **v4 checkpoint 趋势** | soft_landing_bonus: cp40%=127→cp60%=279→cp100%=431；score: cp90%=158→cp100%=89.6 |

### Behavior Diagnosis

**v8 的行为模式：踌躇悬停 + 偶尔着陆但无法固化。**

- `len=759.9, terminated=9/20, truncated=11/20` — 大量超时截断，说明 agent 长期悬停不降落。
- `soft_landing_bonus active_rate=3.4%` — 着陆信号极度稀疏。v4 在 proximity=40 驱动下 active_rate 达到 23.5%，而 v8 的 30 梯度太弱，agent 很少到达垫面触发接触奖励。
- checkpoint 趋势揭示了**学习崩溃**：cp50-60% 时 agent 偶然发现了着陆（soft_landing_bonus=40→80.7, score=36→87），但 cp70% 后行为坍塌（bonus→15, score→-67），最终 cp100% 回到 hover 模式。这是典型的「发现-遗忘」循环：稀疏奖励导致学到的行为不稳定，随着探索率下降，agent 退回到安全的悬停策略（proximity_progress 稳定贡献 ~25/episode 无需冒险着陆）。

**根因**：binary contact gate（`both_contact = legs[0]>0.5 AND legs[1]>0.5`）让 soft_landing_bonus 对 agent 几乎是不可见的 — 97% 的 step 该组件为 0。agent 无法建立「接近垫面→接触→奖励」的因果链。

### 修改方案

**L2 — 将 soft_landing_bonus 的接触门从 binary 改为 graduated（三级：0 / 0.5 / 1.0）。**

- 当前：`both_contact ∈ {0, 1}`，必须双腿同时接触才触发。
- 修改：`contact_level = 0.5·left + 0.5·right ∈ {0, 0.5, 1.0}`，单腿接触给一半奖励。
- 这创建了一个**渐进式学习梯度**：单腿接触（更频繁）→ 双腿接触（最终目标）。agent 先学会「触垫有奖励」，再细化到「双腿着垫奖励更高」。
- 配合微调：proximity 30→35（略微增强接近引导，介于 v8 的过弱和 v4 的 40 之间），soft_landing_bonus coeff 4.0→5.0（补偿单腿接触时减半的 per-step max）。

这个修改在所有历史迭代中从未被尝试过（v1-v8 全部使用 binary contact gate），是实质性的数学形态改变。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v9 — Graduated contact gate for soft_landing_bonus.

    Diagnosis from v8:
    - soft_landing_bonus active_rate = 3.4% — far too sparse for reliable learning.
      The binary gate (both legs must contact simultaneously) means the agent almost
      never experiences the landing reward signal.
    - Checkpoint trend shows the agent discovers landing mid-training (cp50-60%)
      but then forgets it (cp70-100%) — classic sparse-reward collapse as exploration
      decays. Agent retreats to safe hovering where proximity_progress gives steady
      ~25/episode without risk.
    - v4 proved the skeleton works (score=135, 19/20 landed) but needed proximity=40
      to force approach. At proximity=30 the approach gradient is too weak.

    Single semantic change (L2):
    - soft_landing_bonus: binary both_contact → graduated contact_level.
      Single-leg contact now earns half credit (0.5×), creating a stepping-stone
      gradient: "touch the pad" → "land with both legs." This directly addresses
      the sparsity problem by increasing the effective active_rate.
    - Proximity nudged 30→35 and bonus coeff 4.0→5.0 to support the same semantic
      direction (making landing more discoverable and rewarding).
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 35.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: graduated contact for progressive learning ----------
    left_contact = 1.0 if next_obs[6] > 0.5 else 0.0
    right_contact = 1.0 if next_obs[7] > 0.5 else 0.0
    contact_level = 0.5 * left_contact + 0.5 * right_contact   # 0.0, 0.5, or 1.0

    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 5.0 * contact_level * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```