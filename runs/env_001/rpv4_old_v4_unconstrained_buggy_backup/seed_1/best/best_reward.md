---

## 诊断报告

### 1. Evidence

| 来源 | 关键数据 |
|---|---|
| **iter 3 final eval** | score=128.44 (↑ from 67.23), len=573.6, terminated=12/20, best=266.66, range=[-35.5, 266.7] |
| **iter 3 组件** | soft_landing_bonus ep_sum=535.4, **magnitude_share=91.6%**, active_rate=5.8% |
| **soft_landing_bonus 演化** | iter1: 8.0 (14.6%) → iter2: 300.6 (85.4%) → iter3: **535.4 (91.6%)** — 持续膨胀 |
| **active_rate 演化** | iter1: 0.3% → iter2: 3.9% → iter3: **5.8%** — agent 越来越频繁触发 bonus |
| **checkpoint 趋势** | soft_landing_bonus: cp30%=260 → **cp60%=1592 (暴胀!)** → cp70%=751 → cp100%=576 — 经典 exploit 尖峰 |
| **proximity_progress** | 稳定 32-37 ep_sum，但 magnitude_share 从 82.9% 萎缩到 **7.6%** — 被淹没 |
| **attitude_penalty** | 稳定，magnitude_share 仅 0.7% |

### 2. Behavior Diagnosis

**核心问题：soft_landing_bonus 的 transition-based 设计被物理弹跳 exploit。**

虽然 iter 2 引入了 `0→1` transition 机制、iter 3 加了 binary threshold，但 agent 学会了**反复弹跳**：接触垫面（bonus 触发）→ 弹起离开（contact 归零）→ 再次落下接触（bonus 再次触发）。每 episode 平均触发 ~33 次（5.8% × 573 步 ≈ 33 次），每次 ~16 分，累计 535 分。

checkpoint 中 cp60% 暴胀到 1592、随后回落的尖峰模式，是典型的「agent 发现 exploit → 训练部分纠正」循环。这种不稳定训练严重阻碍策略收敛。

**proximity_progress 被彻底淹没**：绝对数值稳定（~32），但相对份额仅 7.6%。agent 主要被弹跳奖励驱动，而非被「接近目标」驱动。

**当前策略行为**：agent 在接近目标后会反复弹跳而非稳定停靠。terminated=12/20（60%）说明仍有 40% 的 episode 未能成功停靠就被截断。

### 3. 干预方案

**L2 修改**：将 `soft_landing_bonus` 从 transition-based event（奖励接触事件）改为 **continuous per-step quality reward**（奖励保持在良好接触状态）。这从根本上消除弹跳激励：弹跳意味着失去接触 → 失去奖励；稳定停靠意味着持续获得奖励。

配套 **L1 调整**：`proximity_progress` 系数 30→40，在 landing bonus 不再爆炸的前提下，强化接近引导信号。

| 组件 | 旧 | 新 | 理由 |
|---|---|---|---|
| soft_landing_bonus | transition-based, coeff=30 | continuous per-step, coeff=4.0 | 奖励「保持」而非「事件」，消除弹跳 exploit |
| proximity_progress | coeff=30 | coeff=40 | 重新平衡，强化接近阶段引导 |
| attitude_penalty | 不变 | 不变 | 工作正常 |

预期：soft_landing_bonus active_rate 会升至接触阶段的 ~100%（因为接触时每步都奖励），但 magnitude_share 会回落到合理范围（~50-60%），不再出现 cp60% 式的暴胀尖峰。

---

## 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v4 — Fixes soft_landing_bonus multi-fire exploit.

    Changes from v3:
    - soft_landing_bonus: transition-based event → continuous per-step quality reward.
      The agent is now rewarded for STAYING in safe contact (settling), not for the
      contact EVENT. This eliminates the bouncing exploit: bouncing means losing
      contact = losing reward. Settling = sustained reward until episode terminates.
    - Coefficient: 30.0 (one-shot) → 4.0 (per-step), bounded by episode termination.
    - proximity_progress: 30.0 → 40.0 to strengthen approach guidance now that the
      landing bonus no longer dominates.
    - attitude_penalty unchanged.
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 40.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: continuous per-step reward for safe contact ----------
    # Only when both legs are firmly on the pad (binary gate)
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    # Quality gates — soft thresholds preserve gradient for learning
    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 4.0 * both_contact * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```