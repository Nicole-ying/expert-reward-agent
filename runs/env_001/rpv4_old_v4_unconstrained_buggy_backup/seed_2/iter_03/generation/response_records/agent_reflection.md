# Response Record

## 诊断报告

### evidence

| 数据点 | iter_1 | iter_2 | 含义 |
|---|---|---|---|
| score | **171.41** | 57.23 | iter_2 崩溃式下降，-114 分 |
| best | 171.41 | — | 最佳分距目标 200 仅差 ~14% |
| len | 781 | 921 | iter_2 更长，agent 在徘徊 |
| terminated | 7/20 | 4/20 | 成功着陆率从 35% 降到 20% |
| contact_guidance ep_sum | 252.86 | **619.18** | 接触奖励膨胀 2.4x，但得分暴跌 |
| contact_guidance active_rate | 68% | 48.4% | 更稀疏却总奖励更大 → 被 exploit |
| contact_guidance mag_share | 99.2% | 99.1% | 两轮都几乎独占 |
| progress ep_sum | 0.65 | 2.54 | progress 提升了但不足以拯救总分 |

**iter_2 checkpoint 趋势关键信号：**
- cp50%: score=111.3 → cp70%: 147.9 → cp90%: 80.8 → cp100%: 63.4（先升后降，经典 exploit 模式）
- contact_guidance ep_sum: cp50%=287 → cp100%=551（单调膨胀）
- score 在 cp70% 见顶后持续下滑，说明 agent 学会了刷 contact_bonus 而非完成任务

**iter_1 checkpoint 趋势（对比）：**
- score 从 cp70%=105.8 一路升至 cp100%=223.9（稳定上升）
- contact_guidance 在 cp70% 跃升后保持稳定，未膨胀

### behavior_diagnosis

**iter_2 的策略已学坏。** iter_1 的 dense contact reward（`0.3 * contact_val * proximity * velocity_ok`）有效引导 agent 达到 171 分。iter_2 试图用 transition-based bonus（`5.0 * contact_gained`）解决 contact_guidance 占比过高的问题，但过渡奖励被 exploit：agent 学会了反复建立→断开接触来刷取 5.0 的 transition 奖励。证据是 contact_guidance ep_sum 从 252 膨胀到 619，但实际任务得分从 171 暴跌到 57。

**核心错误：transition bonus 太大（5.0）且可被循环触发。** agent 不再追求稳定着陆，而是在平台附近振荡接触。

**正确方向：回到 iter_1 的 dense 结构。** iter_1 证明 dense contact reward 能有效引导行为且不被 exploit。问题仅在于量级平衡——contact_guidance 占比 99%+ 使得 progress 和速度约束形同虚设。

### 修改方案（L1：仅调系数）

1. **contact_guidance 权重 0.3 → 0.08**：大幅削弱，让 progress 有存在感
2. **progress 权重 0.5 → 3.0**：显著提升，使距离改善成为有意义的学习信号
3. **velocity_ok 改用 total speed**（保留 iter_2 的合理改进）：同时约束水平和垂直速度，比仅看 vy 更安全
4. **velocity_penalty 和 orientation_penalty** 保持 iter_1 原样

这是纯 L1 修改——不改数学形态，只调系数 + 一处微小的 velocity_ok 变量替换（从 `abs(vy)` 改为 `speed`）。

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
    contact_guidance = 0.08 * contact_val * proximity_factor * velocity_ok

    # ── combine all components ──
    progress_reward = 3.0 * progress

    total_reward = progress_reward + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": progress_reward,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return float(total_reward), components
```
