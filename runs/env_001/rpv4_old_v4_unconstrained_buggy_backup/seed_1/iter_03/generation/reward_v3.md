Now I have a complete picture. Let me compile my analysis.

---

## 诊断报告

### 1. Evidence

| 来源 | 关键数据 |
|---|---|
| **iter 2 final eval** | score=67.23 (目标200), len=561, terminated=13/20, soft_landing_bonus ep_sum=300.56 (85.4% magnitude), active_rate=3.9% |
| **iter 2 checkpoint** | soft_landing_bonus: cp70%=1.8 → cp80%=76.7 → **cp100%=322.4** (爆炸增长); active_rate: cp70%=1.1% → cp100%=**4.5%** |
| **iter 1 vs iter 2** | soft_landing_bonus 从 8.0 (state-based, active 0.3%) 跳到 300.6 (transition-based, active 3.9%) — 37x 增长 |
| **cross-check** | 如果 bonus 真正 one-shot，active_rate 应为 ~1/561≈0.18%；实际 3.9%≈22次/ep — **确认多次触发** |
| **contact 信号性质** | `obs[6]`/`obs[7]` 声明为"连续 1.0 或 0.0"，但物理引擎可能产生中间值，`obs[6]*obs[7]` 在接触变化时产生连续渐变 |

### 2. Behavior Diagnosis

**核心问题：soft_landing_bonus 的 transition 检测被 exploit。**

v1（iter 1）使用 state-based bonus（每步在垫上就给奖励），agent 学会了"sit and collect"——在垫上停留持续收奖励，导致 training score 虚高（checkpoint cp80%=164.9）但 eval 崩盘（-2.98）。

v2（iter 2）试图通过 transition-based 修复：`(1 - was_both_contact) * is_both_contact`。但这里有一个致命漏洞：**如果 `obs[6]` 和 `obs[7]` 不是严格二值（0/1），而是连续值**，那么 `was_both_contact = obs[6] * obs[7]` 会在接触力度变化时产生中间值（如 0.72→0.90），导致 `new_contact_event` 在每个接触"加深"的 step 都部分触发（~0.25）。这解释了：
- active_rate 4.5%（~25次/ep）而非预期的 0.18%
- cp100% 时 ep_sum 爆发到 322，远超 cp80% 的 77
- 外部得分仅 67/200——agent 学会了频繁制造"接触改进"事件而非真正稳定着陆

**策略行为模式**：agent 学会触碰垫面、轻微弹跳、再次接触，循环获取 bonus 而非稳定停靠。这解释了 terminated=13/20（多数最终终止但可能是不稳定着陆）和长 episode（561步）。

### 3. 修改方案

**单一语义修改：让 soft_landing_bonus 成为真正的 one-shot 事件。**

- **L2**：对 contact 做阈值二值化 (`> 0.5`)，确保 transition 只有严格的 0→1 跳变
- **L1**：系数从 20→30，补偿真正 one-shot 后单次奖励的合理量级（预期 ~15-27，与 proximity_progress ~32 平衡）

两个改动服务于同一语义：消除多次触发 exploit，让 bonus 回归"首次双足接触"的设计意图。

---

## 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v3 — Fixes soft_landing_bonus multi-fire exploit with binary contact threshold.

    Changes from v2:
    - Contact detection: continuous product → thresholded binary (0.5 cutoff)
      Prevents partial transition events when contact values fluctuate.
    - Coefficient: 20.0 → 30.0 to maintain meaningful one-shot landing incentive
      now that the bonus truly fires only once per episode.
    - proximity_progress and attitude_penalty unchanged (both healthy).
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: TRUE one-shot reward on first safe contact ----------
    # Binary threshold prevents exploit from continuous contact-value fluctuations
    was_both_contact = 1.0 if (obs[6] > 0.5 and obs[7] > 0.5) else 0.0
    is_both_contact  = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    new_contact_event = (1.0 - was_both_contact) * is_both_contact  # 1.0 ONLY on 0→1 transition

    # Quality gates — soft thresholds preserve gradient for learning
    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 30.0 * new_contact_event * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```