现在我有了完整的图景。让我整理诊断和干预方案。

---

## 诊断报告

### 1. Evidence（关键证据）

**Checkpoint 趋势揭示的「发现时刻」：**
- cp10%-cp60%：contact_guidance 几乎为零（0.0~9.4），agent 在黑暗中摸索，score 为负（-33 到 -132）
- **cp70%：contact_guidance 骤升至 248.8，score 从 -15.9 跃升至 105.8** — agent 发现了「接触=奖励」
- cp70%→cp100%：contact_guidance 从 270 下降到 167，同时 score 从 105 上升到 224 — agent 在加速着陆、减少滞留

**最终评估的症候群：**
| 指标 | 值 | 含义 |
|---|---|---|
| contact_guidance magnitude_share | **99.2%** | 所有其他信号被淹没 |
| terminated | **7/20** | 65% 的 episode 无法触发 settled 终止 |
| len | **781** | 长 episode，大量时间消耗在平台附近 |
| score range | [106, 282] | 方差极大，取决于是否成功终止 |

### 2. Behavior Diagnosis（行为诊断）

Agent 学到的是一个**局部最优策略**：飞到平台附近 → 保持低速 + 部分/间歇接触 → 每步收割 0.3~0.6 的 contact_guidance 奖励 → **不触发 settled 终止，无限滞留**。

证据链：
- 高 contact_guidance 累积（252.86）÷ 长 episode（781）÷ 中等 active_rate（68%）= 长时间处于「接触+近目标+低vy」状态
- 仅有 7/20 触发终止 → 大部分 episode agent 满足于「够好」状态而非「完全 settled」
- 这不是 agent 找不到平台，而是**找到了但不停下**

根本原因：`contact_guidance` 是**状态占有型奖励**（state-occupancy reward）。只要 agent 维持在「接触+近+低速」状态，就能持续获得奖励，形成一个 **reward plateau**。

### 3. Signal Completeness（信号完备性）

四类必要信号全部存在：proximity ✓、velocity ✓、orientation ✓、contact ✓。**不缺信号，缺的是 contact 信号的结构**：当前是占有型（鼓励滞留），需要改为进步型（鼓励达成后终止）。

### 4. Selected Level：**L2**

理由：
- 仅调系数（L1）无法解决「状态占有→plateau」这一结构性问题。即使把 contact_guidance 从 0.3 降到 0.05，agent 仍会滞留（只是每步收割更少，但累积仍可观）
- 差距 14.5% > 10%，L2 不被禁止
- 这是接触信号**数学形态**的改变：从 `Φ(s)` → `Φ(s') - Φ(s)`（势能差），语义上从「状态奖励」变为「改进奖励」

### 5. Selected Intervention（目标组件及修改）

**目标组件：`contact_guidance`**

将 contact_guidance 从**状态占有型奖励**改为**势能差奖励**（potential-based shaping）：

```
旧：contact_guidance = 0.3 × contact_val × proximity × velocity_ok    [每步占有即奖励]
新：contact_guidance = 100.0 × (Φ(next) - Φ(prev))                     [只奖励改进]
```

其中 `Φ = contact_val × proximity × velocity_ok`。

**配套调整**：progress 权重 0.5 → 2.0（同一语义方向：强化「到达目标」的信号以弥补 contact 总奖励的下降）。

### 6. Falsifiable Hypothesis（可证伪假设）

> 将 contact 从占有型改为势能差型，将消除 reward plateau。Agent 无法通过滞留收割奖励，必须持续改进接触状态。一旦达成完全接触+近目标+零速度（Φ≈2.0），contact_guidance 归零，agent 没有理由继续停留 → 自然触发 settled 终止。

**预期可证伪信号：**
- ✅ contact_guidance 的 magnitude_share 从 99% 降至 < 50%
- ✅ terminated 比例从 7/20 上升
- ✅ episode length 下降
- ❌ 如果 contact_guidance 骤降但 score 也骤降（agent 找不到平台），说明势能系数过高/过低

### 7. Expected Next Round（下轮预期行为）

Agent 将学到：**快速接近 → 减速 → 建立接触 → 稳定 → 终止**。不再滞留。预期 score 接近或超过 200，terminated 率 > 50%。

---

## 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to target (target at origin (0,0)) ---
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # --- velocity components ---
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # --- orientation penalty: body angle and angular velocity ---
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # --- contact guidance: potential-based difference reward ---
    # Reward IMPROVEMENT in contact state, not occupancy of it.
    # This eliminates the plateau where the agent can hover near the platform
    # collecting rewards indefinitely without fully settling.
    def contact_potential(dist, vy_val, contact_sum):
        """Potential function: higher when closer, slower, and more legs on ground."""
        prox = max(0.0, 1.0 - dist / 5.0)
        v_ok = max(0.0, 1.0 - abs(vy_val) / 0.5)
        return contact_sum * prox * v_ok

    contact_prev = float(obs[6] + obs[7])
    contact_next = float(next_obs[6] + next_obs[7])

    pot_prev = contact_potential(dist_prev, obs[3], contact_prev)
    pot_next = contact_potential(dist_next, next_obs[3], contact_next)

    contact_guidance = 100.0 * (pot_next - pot_prev)

    # --- combine all components ---
    # Progress weight increased from 0.5 to 2.0 to strengthen the approach signal
    # now that contact_guidance no longer provides a constant occupancy reward.
    total_reward = (2.0 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 2.0 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```