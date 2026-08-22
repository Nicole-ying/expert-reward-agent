# Response Record

现在我对数据有了全面把握。让我整理诊断并设计新奖励函数。

---

## 诊断报告

### Evidence

| 信号 | 数据 |
|---|---|
| `contact_guidance` magnitude_share | **99.2%** — 完全主导奖励信号 |
| `contact_guidance` ep_sum_mean | 252.86 vs 总分 171.41（contact 奖励自己超过了总分！） |
| `progress` signed_share | 0.3%，ep_sum_mean 仅 0.65 — 几乎无作用 |
| `velocity_penalty` signed_share | 0.1%，ep_sum_mean -0.18 — 可忽略 |
| `orientation_penalty` signed_share | 0.5%，ep_sum_mean -1.20 — 可忽略 |
| Checkpoint cp60%→cp70% | contact_guidance 从 **0.00 → 248.83**（暴增），经典 exploit 信号 |
| 终止情况 | 仅 7/20 terminated，13/20 truncated；ep_len=781（极长） |

### Behavior Diagnosis

**agent 学到了什么：** 在 cp70% 左右，agent 突然发现了 `contact_guidance` 的奖励机制，然后全力 exploit 它。原始代码中 `proximity_factor = max(0, 1 - dist/5.0)` 条件太过宽松——在目标周围半径 5 单位的广大区域内都能获得接触奖励。agent 的策略变为：在目标附近徘徊，间歇性让双腿触地刷 contact 奖励，而非高效完成着陆。

**为什么不健康：**
1. `contact_guidance` 占 signed_share 的 99.2%，其他三个组件的信号完全被淹没，RL 无法从它们学到任何东西
2. 781 步的极长 episode 和 65% 的截断率表明 agent 在"刷分"而非完成任务
3. 进度奖励（progress）几乎为零——agent 没有动力快速接近目标，只是在目标附近磨蹭

### 干预方向

1. **contact_guidance：大幅收紧条件，降低有效量级** — proximity 阈值从 5.0 → 0.3（面积缩小 ~278 倍），速度阈值从仅检查 vy → 检查总速度且更严格
2. **progress：提升为主驱动信号** — 权重从 0.5 → 3.0，使接近目标的奖励在量级上与其他组件可比
3. **velocity_management：拆分并加强** — 加入基础速度惩罚 + 近地下降速度惩罚，防止高速撞击
4. **orientation_penalty：适度加强** — 从 -0.05/0.02 → -0.15/0.05

---

## 新奖励函数

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── Distance to target (origin at 0, 0) ──
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # ── 1. Progress: PRIMARY driver toward target ──
    # potential-based: sum over episode = dist_start - dist_end
    progress = dist_prev - dist_next          # >0 when approaching target
    progress_reward = 3.0 * progress          # was 0.5 → 6x boost

    # ── 2. Velocity management ──
    vx = next_obs[2]
    vy = next_obs[3]
    speed_sq = vx ** 2 + vy ** 2
    speed = speed_sq ** 0.5

    # 2a. General speed penalty — encourages efficiency, prevents oscillation
    speed_penalty = -0.08 * speed_sq

    # 2b. Hard-landing prevention: penalise downward speed near ground
    #     exponential decay with distance → strong only when close to target
    near_ground = 2.718281828 ** (-3.0 * dist_next)
    descent_penalty = -0.8 * max(0.0, -vy) * near_ground

    velocity_management = speed_penalty + descent_penalty

    # ── 3. Orientation stabilisation ──
    orientation_penalty = -0.15 * abs(next_obs[4]) - 0.05 * (next_obs[5] ** 2)

    # ── 4. Contact guidance: tight, conditional, modest ──
    contact_val = float(next_obs[6] + next_obs[7])   # 0.0 .. 2.0

    # Tight proximity: only within 0.3 units of target (was 5.0!)
    proximity_ok = max(0.0, 1.0 - dist_next / 0.3)

    # Tight speed gate: total speed < 0.2 (was vy-only < 0.5)
    speed_ok = max(0.0, 1.0 - speed / 0.2)

    # Modest coefficient 0.3 × tighter gates → much lower effective magnitude
    contact_guidance = 0.3 * contact_val * proximity_ok * speed_ok

    # ── Combine ──
    total_reward = (
        progress_reward
        + velocity_management
        + orientation_penalty
        + contact_guidance
    )

    components = {
        "progress": progress_reward,
        "velocity_management": velocity_management,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```

### 量级预期对比

| 组件 | 旧版 ep_sum 量级 | 新版预期 | 变化 |
|---|---|---|---|
| `progress` | ~0.65 | ~15–25 | **6× 权重** |
| `velocity_management` | ~-0.18 | ~-5 到 -15 | 拆分+加强，成为有意义的信号 |
| `orientation_penalty` | ~-1.20 | ~-5 到 -10 | 适度加强 |
| `contact_guidance` | ~252 | ~2–8 | 条件收紧 ~100× 有效面积，量级回归合理 |

核心变化：**progress 成为主导信号**（驱动 agent 快速飞向目标），contact_guidance 退居为最后触地阶段的微调信号（仅在目标 0.3 单位内、速度 <0.2 时生效），velocity_management 真正起到防止撞击的保护作用。
