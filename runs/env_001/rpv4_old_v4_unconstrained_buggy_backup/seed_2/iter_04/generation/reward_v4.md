## 诊断报告

### evidence

| 数据点 | iter_1 | iter_2 | iter_3 | 趋势 |
|---|---|---|---|---|
| score | 171.4 | 57.2 | **246.4** | ✅ 已超越目标 200 |
| best | 171.4 | 171.4 | **246.4** | ✅ 新 best |
| len | 781 | 921 | **533** | ✅ 显著缩短 |
| terminated | 7/20 | 4/20 | **16/20** | ✅ 80% 成功着陆 |
| contact_guidance ep_sum | 252.9 | 619.2 | 44.1 | ✅ 大幅收敛，不再爆炸 |
| contact_guidance share | 99.2% | 99.1% | **88.9%** | ⚠️ 仍占主导 |
| progress share | 0.3% | 0.5% | **8.6%** | 📈 逐步提升但仍偏弱 |
| velocity_penalty share | 0.1% | 0.0% | 0.4% | 可忽略 |

**Checkpoint 趋势关键信号（iter_3）：**
- 分数从 cp50% 的 45 跃升到 cp60% 的 218，之后稳定在 230~264，趋势健康
- contact_guidance ep_sum 剧烈震荡：cp60%=22.2 → cp70%=6.7 → cp80%=19.1 → cp90%=6.2 → cp100%=23.8。这说明 contact 信号对策略微小变化高度敏感，agent 在「获得接触」和「丢失接触」之间反复
- progress 稳步从 2.0 增长到 4.1，是唯一持续单调上升的组件——它是稳定性的来源
- cp90%→cp100% 分数微降（263.7→257.4），有轻微过拟合迹象

### behavior_diagnosis

策略当前行为模式：**agent 已学会到达目标平台并着陆**（80% terminated，len=533），但 reward 结构仍以「处于接触状态」为主要驱动力（88.9%），而「向目标前进」的导航信号仅占 8.6%。这意味着 agent 主要因为碰巧进入接触区才获得大量奖励，而非因为高效地飞向目标。

健康度评估：**基本健康但 reward balance 可以优化**。contact_guidance 的剧烈震荡（checkpoint 间 3x 波动）表明当前 0.08 系数下，contact 信号在训练后期对策略扰动过于敏感——微小的策略变化导致接触奖励大幅波动。适度降低 contact 权重、同时提升 progress 权重，可以让导航信号在 contact 缺失时提供更稳定的引导，减少对 contact 单一信号的过度依赖。

### 修改方案：L1 权重再平衡

**语义方向**：继续 iter_2→iter_3 的成功方向——将奖励重心从「停留在接触状态」进一步向「高效导航靠近目标」转移。

- `contact_guidance`: 0.08 → **0.05**（降低 37.5%，预期 ep_sum ~27.6，share ~75%）
- `progress`: 3.0 → **5.0**（提升 67%，预期 ep_sum ~6.9，share ~19%）

理由：分数已超目标且 terminated=80%，无需 L2 改公式。这是沿着 iter_1→iter_2→iter_3 已验证有效的方向继续微调（contact ↓, progress ↑），降低 late-stage contact 震荡风险，同时不破坏已学到的着陆行为。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── distance to target (target at origin (0,0)) ──
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # ── velocity components ──
    vx = next_obs[2]
    vy = next_obs[3]
    speed = (vx**2 + vy**2) ** 0.5

    # ── velocity penalty (same as iter_1, proven gentle and effective) ──
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor
    vel_horizontal_penalty = -0.01 * (vx ** 2)
    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # ── orientation penalty ──
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # ── contact guidance: dense reward, only near target + low speed ──
    contact_val = float(next_obs[6] + next_obs[7])   # range 0.0 .. 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    velocity_ok = max(0.0, 1.0 - speed / 0.5)        # both vx and vy must be low
    contact_guidance = 0.05 * contact_val * proximity_factor * velocity_ok

    # ── combine all components ──
    progress_reward = 5.0 * progress

    total_reward = progress_reward + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": progress_reward,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return float(total_reward), components
```