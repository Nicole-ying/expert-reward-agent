# Response Record

---

## 诊断报告

### Evidence

**关键数据对比（iter_1 → iter_2）：**

| 指标 | iter_1 (state-value) | iter_2 (potential-diff) | 变化 |
|---|---|---|---|
| score | 171.41 | 157.07 | **↓14** |
| terminated | 7/20 | 3/20 | **↓4** |
| len | 781 | 926 | **↑145** |
| contact_guidance signed_share | 99.2% | 2.8% | **↓96%** |
| contact_guidance magnitude_share | 99.2% | 99.9% | → |
| contact_guidance ep_sum | 252.9 | 155.5 | ↓ |

**关键信号：signed_share 从 99.2% 暴跌到 2.8%，但 magnitude_share 维持在 99.9%。** 这意味着 contact_guidance 正在产生巨大的正负振荡：每一步的 potential 变化被 100.0 倍放大，产生大量正负交替的奖励尖峰，在绝对值上主导训练，但净值几乎为零。

**Checkpoint 趋势佐证：**
- iter_1 在 cp70% 有一个明确的「发现时刻」：contact_guidance 从 0 骤升至 248.8，score 从 -15.9 跃升至 105.8
- iter_2 没有这样的发现时刻——contact_guidance 从 cp10% 就开始缓慢爬升（24.7→191.9），说明 agent 在持续收到零散的 potential-diff 信号，但没有清晰的「这就是着陆」时刻
- iter_2 的 terminated 从 7→3，长度从 781→926：agent 更少成功着陆，更多超时

### Behavior Diagnosis

当前策略正在经历 **奖励信号污染**：100.0 倍的 potential-diff 使得飞行器的每一步微小位置/速度/接触变化都产生巨大的 ± reward spike。从 signed_share=2.8% 可以推断，这些 spike 在 episode 内大致正负抵消，但其 magnitude 完全淹没了 progress（0.1% magnitude_share）、velocity_penalty 和 orientation_penalty。agent 被噪声主导，无法从 progress 信号中学习清晰的「靠近目标」行为。

结果：agent 着陆率下降（3/20 vs 7/20），episode 变长，整体得分下降。

### Signal Completeness

四个职责组件都存在，但 **contact_guidance 的尺度完全压倒了其他三个**。progress 的 signed_share 接近 0%，意味着 agent 在实际学习中几乎忽略了「向目标靠近」这一基本信号。velocity_penalty 和 orientation_penalty 始终活跃但量级微不足道（合计不到 0.1% magnitude_share），无法起到约束作用。

### Selected Level: **L1**

理由：potential-diff 框架（iter_2 的数学形态）在方向上是正确的——它解决了 iter_1 的 occupancy exploitation 问题（signed_share 从 99.2% 降到 2.8% 证明不再有「赖着不走吃奖励」的行为）。问题纯粹是系数失衡：100.0 倍放大把有效的 difference 信号变成了噪声。只需缩小 contact_guidance、放大 progress，让信号回到同一数量级。

### Selected Intervention

**单一语义修改：重新平衡 contact_guidance 与 progress 的系数**

- `contact_guidance` 乘数：100.0 → **3.0**（降低 33 倍，使其 ep_sum 从 ~155 降至 ~4.7，与 progress 可比）
- `progress` 权重：2.0 → **5.0**（提升 2.5 倍，确保接近目标的主驱动信号不被淹没）

### Falsifiable Hypothesis

如果假设成立：contact_guidance 的 magnitude_share 将从 99.9% 显著下降（预计降至 30-60%），progress 的 signed_share 将转正并占据主导。agent 将更多地被「向目标靠近」驱动，而非被 contact potential 的微小波动支配。结果：terminated 率回升（≥5/20），score 回升至 ≥170 并可能突破 best=171.41。

如果假设失败（score 继续下降或不变）：说明 potential 函数本身的结构（contact_sum × prox × v_ok 三因子乘积导致空中阶段恒为零）需要 L2 修改，需要将 proximity 从乘积中解耦。

### Expected Next Round

agent 将被 progress 信号稳定地驱向目标平台，同时在接近目标后由 contact_guidance（3.0 倍 potential-diff）引导完成轻柔触地。episode 长度预计缩短（更高效的接近），terminated 率回升。contact_guidance 的 magnitude_share 降至合理水平（不再垄断训练梯度），orientation_penalty 和 velocity_penalty 的相对影响力自然提升。

---

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

    # Multiplier reduced from 100.0 to 3.0 to stop dominating training with
    # large-magnitude oscillations (was 99.9% magnitude_share, 2.8% signed_share).
    contact_guidance = 3.0 * (pot_next - pot_prev)

    # --- combine all components ---
    # Progress weight increased from 2.0 to 5.0 to strengthen the approach signal
    # now that contact_guidance no longer monopolizes the gradient.
    total_reward = (5.0 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 5.0 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```
