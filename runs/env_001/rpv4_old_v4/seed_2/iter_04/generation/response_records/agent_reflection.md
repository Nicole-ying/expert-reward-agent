# Response Record

---

## 诊断报告

### 1. Evidence（关键证据）

**三代 contact_guidance 演化揭示的崩塌路径：**

| 指标 | iter_1 (state-value) | iter_2 (potential-diff, ×100) | iter_3 (potential-diff, ×3) |
|---|---|---|---|
| contact ep_sum | 252.9 | 155.5 | **3.8** |
| contact active_rate | 68.0% | 81.0% | **0.9%** |
| contact magnitude_share | 99.2% | 99.9% | 32.5% |
| **score** | **171.41** | **157.07** | **-73.59** |

**iter_3 checkpoint 趋势佐证：**
- Score 全程为负：cp10%=-124.7 → cp90%=-44.3 → cp100%=-60.2（略有学习但从未转正）
- contact_guidance active_rate 全程 < 8%，在 cp100% 仅 7.2%
- progress ep_sum 平坦在 5.5-6.2（≈1.2 单位距离改进），远不足以到达平台
- 12/20 集在 150 步内以 score < -50 提前终止 → **agent 在快速坠毁/飞出边界**

**为什么 potential-diff 在 iter_3 彻底死亡：** 势能函数 `contact_sum * prox * v_ok` — 当 `contact_sum=0`（双腿离地），整个势能为零。`pot_next - pot_prev` 仅在接触状态变化时非零（离散事件）。iter_2 靠 ×100 放大浮点噪声维持 81% active，iter_3 降到 ×3 后，真正的稀疏性暴露：active_rate 仅 0.9%。

### 2. Behavior Diagnosis

当前策略学到的是**无引导坠落**：progress（5.0×距离改进）是唯一正向信号，但幅度太小（ep_sum≈6，即改进约 1.2 距离单位），速度/姿态惩罚全程负向压制。agent 得不到任何关于"平台在哪、接触是什么"的有效反馈，只能随机探索，大部分 episode 迅速 crash。这**极其不健康**。

### 3. Signal Completeness

| 职责 | 状态 | 问题 |
|---|---|---|
| goal_proximity (progress) | ✅ 有信号 | 信号幅度不足单独引导 |
| velocity_management | ✅ 有信号 | 正常 |
| orientation_stabilization | ✅ 有信号 | 正常 |
| **safe_contact** | ❌ **死亡** | 0.9% active，agent 不知道接触=好 |

safe_contact 是着陆任务最关键的终端引导信号，它死了，整个奖励函数就失去了"终点在哪"的定义。

### 4. Selected Level: **L2**

理由：需要改变 contact_guidance 的数学形态，从势能差分回到状态值奖励。iter_1 已证明 state-based 公式有效（score=171，距目标 200 仅差 29 分）。这不是凭空发明，而是回到已验证的方案。

### 5. Selected Intervention

**目标组件：`contact_guidance`**

从势能差分 `3.0 * (pot_next - pot_prev)` 回退到 iter_1 的状态值公式：
```
contact_guidance = 0.3 * contact_val * proximity_factor * velocity_ok
```
同时保留 progress 权重 5.0（比 iter_1 的 0.5 更强，提供更积极的逼近信号）。

### 6. Falsifiable Hypothesis

状态值奖励是**稠密的 occupancy 信号**：只要 agent 处于"近平台 + 低速 + 有接触"状态，每步都获得正奖励。这解决了 potential-diff 的稀疏性问题。iter_1 证明 agent 能借此发现"接触=好"并学会着陆（cp70% 时 contact 从 0 骤升至 248.8，score 从 -15.9 跃升至 105.8）。

**可能的风险**：occupancy 奖励可能诱导 agent 在平台上滞留（hover and collect）。但 iter_1 的 len=781 和 7/20 terminated 表明 agent 最终会 settle，且 score=171 已接近目标。即便有此副作用，也比当前 -73.59 的 crash 模式好得多。

### 7. Expected Next Round

- contact_guidance active_rate 回升至 50-70%
- contact_guidance ep_sum 回升至 100-250 范围（强正向引导）
- score 回升至 +100 以上，agent 重新学会飞向并接触平台
- len 可能较长（agent 在平台上停留），但 terminated 比例应回升
- 若 score 回到 150+，下一轮可 L1 微调权重平衡 speed vs fuel

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

    # --- contact guidance: state-based occupancy reward ---
    # Reverted from potential-based difference (iter_2/iter_3) back to the
    # state-value formula proven in iter_1 (score=171). The potential-diff
    # approach is fundamentally broken for discrete contact signals:
    # contact_sum * prox * v_ok = 0 whenever legs are off the ground,
    # making pot_next - pot_prev = 0 in >99% of steps (active_rate=0.9%).
    #
    # State-based reward gives dense positive signal whenever the agent is
    # near the platform, slow, and has legs touching — guiding it to
    # discover and sustain contact.
    contact_val = float(next_obs[6] + next_obs[7])   # range 0.0 .. 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    velocity_ok = max(0.0, 1.0 - abs(vy) / 0.5)
    contact_guidance = 0.3 * contact_val * proximity_factor * velocity_ok

    # --- combine all components ---
    # Progress weight kept at 5.0 (from iter_3) to provide stronger approach
    # signal than iter_1's 0.5, while contact_guidance remains the dominant
    # terminal guidance channel.
    total_reward = (5.0 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 5.0 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```
