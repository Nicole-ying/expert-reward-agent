Now I have all the data I need. Let me compile the full diagnosis.

---

## 诊断报告

### 1. Evidence（关键证据）

**iter_6 → iter_7 的改善原因**：

iter_6 使用 transition-based contact（`100.0 * contact_gain * proximity_factor`），导致 bouncing exploitation：agent 通过在平台上反复弹跳来重复获取接触转换奖励，contact ep_sum 暴涨至 9810（active 仅 12.1%），但实际任务完成度差（score=101.8）。

iter_7 回退到 state-value contact，但新增了两个关键门控：
- **速度门控从仅 |vy| 扩展为 total speed**（`(vx²+vy²)⁰·⁵ < 0.5`），堵住了水平滑行漏洞
- **新增角度门控**（`|angle| < 0.3`），堵住了倾斜站立的漏洞

这三个门控（proximity + speed + angle）同时收紧，消除了 exploitation，使 contact ep_sum 从 9810 降至 179.7，但 score 从 101.8 飙升至 237.76。**证据：contact 的 ep_sum 下降了 98%，但 terminated 从 8/20 升至 14/20，len 从 786 降至 517——agent 学会了真正 settle 而非 exploit。**

**Checkpoint 趋势揭示的核心动态**：

| 阶段 | contact ep_sum | score | 解读 |
|:---|:---|:---|:---|
| cp80% | 354.4 | 140.1 | agent 在平台上 linger，大量收集 contact 奖励 |
| cp90% | 23.6 | **265.9** | agent 学会快速 settle，contact 累积少但终止快 |
| cp100% | 70.5 | 257.5 | 轻微回退，但仍保持 settle 策略 |

**关键洞察：低 contact 累积 + 快速终止 = 高 score。** 当 agent linger 收集 contact 时（cp80%），score 反而不高；当它快速 settle（cp90%），contact 骤降但 score 达到峰值。

**contact_guidance 演化史**：

| iter | 公式类型 | ep_sum | active% | score |
|:---|:---|:---|:---|:---|
| 1 | state-value (vy gate only) | 252.9 | 68.0% | 171.4 |
| 5 | state-value (vy gate only) | 328.9 | 66.4% | 162.3 |
| 6 | transition-based (×100) | 9810.3 | 12.1% | 101.8 |
| **7** | **state-value (speed+angle gate)** | **179.7** | **60.9%** | **237.8** |

### 2. Behavior Diagnosis

当前策略已学会**接近目标 → 减速 → 摆正姿态 → 双脚接触 → 触发 settle 终止**的完整序列。14/20 成功终止，len=517，score=237.76（超过目标 200）。

但不健康的地方：
- **contact_guidance 占 98.9% magnitude share**，其他三个组件（progress 0.4%、velocity 0.1%、orientation 0.6%）实质上是装饰性信号。agent 在 approach 阶段几乎完全盲飞，只在接近平台且满足所有 gate 条件时才获得有意义的反馈。
- **6/20 truncated**（30% 失败率），这些 episode 中 agent 未能触发 settle 条件，可能是 approach 阶段缺乏引导导致无法到达目标区域。
- **Checkpoint 间 score 剧烈波动**（cp70%: 84.2 → cp90%: 265.9），说明 multiplicative gating 创造的奖励曲面非常陡峭——四因子相乘，任一不满足奖励归零。

### 3. Signal Completeness

- `goal_proximity`（progress）：**存在但权重过低**（0.4% share），approach 阶段无有效引导
- `velocity_management`：存在且活跃，但量级微小
- `orientation_stabilization`：同上
- `safe_contact`：信号精准但过于 dominant
- **不缺信号类型，缺的是信号平衡**

### 4. Selected Level: **L1**

理由：
- score=237.76 已超过目标 200，不宜冒险做 L2
- iter_5 尝试 L1（progress 0.5→1.0）时用了 looser gates 导致 score 下降，但 iter_7 的 tighter gates 改变了 reward landscape，适度的 progress 增强可能帮助而非伤害
- 当前最明确的改进方向是**信号再平衡**，不改变数学形态

### 5. Selected Intervention

**目标组件：`contact_guidance` 和 `progress`**

- `contact_guidance`：系数 0.3 → 0.25（降低 17%）
- `progress`：权重 0.5 → 0.8（提高 60%）

这不是两个独立修改，而是服务于同一语义：**减少 contact 的绝对主导地位，增强 approach 阶段的引导信号**。两个参数联动调整，使 progress 从"装饰性"变为"有意义"（预期从 0.4% share 升至 ~2-3%），同时 contact 仍保持主导但略有收敛。

### 6. Falsifiable Hypothesis

**如果假设正确**：progress 提供更好的 approach 引导 → 更多 episode 能到达目标区域 → truncated 从 6/20 下降 → score 保持或微增，且 checkpoint 间波动减小。

**如果假设错误**：progress 增强导致 agent 急于冲向目标（rush-and-crash），terminated 下降，score 回落至 ~160 以下。此时应回退 progress 权重，考虑从 gate 宽松度入手。

### 7. Expected Next Round

- score：230-250 区间（保持或微增）
- terminated：≥15/20
- contact_guidance magnitude_share：从 98.9% 降至 90-95%
- progress magnitude_share：从 0.4% 升至 2-4%
- checkpoint 趋势更平滑，contact 不再出现 cp80% 那种 354 的尖峰

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

    # --- contact guidance: state-value with comprehensive settling gates ---
    # Three gates (proximity + speed + angle) must all be satisfied simultaneously
    # to receive contact reward. This ensures reward only fires during genuine settling.
    # L1 change: coefficient 0.3 → 0.25 to slightly reduce contact dominance (was 98.9%
    # magnitude share), giving relatively more room for progress to guide the approach phase.
    contact_val = float(next_obs[6] + next_obs[7])   # 0.0, 1.0, or 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    speed = (vx**2 + vy**2) ** 0.5
    velocity_ok = max(0.0, 1.0 - speed / 0.5)       # speed < 0.5 for any reward
    angle_ok = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)  # |angle| < 0.3 for any reward
    contact_guidance = 0.25 * contact_val * proximity_factor * velocity_ok * angle_ok

    # --- combine all components ---
    # L1 change: progress weight 0.5 → 0.8.
    # At 0.5, progress contributed only 0.4% magnitude share — essentially decorative.
    # At 0.8, progress provides meaningful approach-phase guidance while contact_guidance
    # (now at 0.25) remains the dominant terminal signal. The goal is to give the agent
    # more feedback during the blind approach phase, helping the 6/20 truncated episodes
    # that never reach the contact zone.
    total_reward = (0.8 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 0.8 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```