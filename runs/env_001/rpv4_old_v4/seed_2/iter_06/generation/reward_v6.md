---

## 诊断报告

### 1. Evidence（关键证据）

**五轮迭代演化（contact_guidance 为核心线索）：**

| iter | contact 公式 | contact ep_sum | contact active% | contact signed% | contact mag% | score |
|:---:|---|---:|---:|---:|---:|---:|
| 1 | state-value (0.3×) | 252.9 | 68.0% | 99.2% | 99.2% | **171.41** |
| 2 | potential-diff (×100) | 155.5 | 81.0% | 2.8% | 99.9% | 157.07 |
| 3 | potential-diff (×3) | 3.8 | 0.9% | 32.5% | 32.5% | -73.59 |
| 4 | state-value (progress=5) | 0.25 | 0.8% | 3.7% | 3.7% | -52.53 |
| 5 | state-value (progress=1) | 328.9 | 66.4% | 99.2% | 99.2% | **162.29** |

**iter_5 checkpoint 趋势 — 经典的 exploitation 模式：**

| | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---|---|---|---|---|---|
| score | 189.9 | -8.3 | **234.7** | 218.7 | 170.4 | **157.2** |
| contact_guidance | 68.3 | 0.0 | 27.9 | 95.1 | 249.2 | **331.9** |

Score 在 cp70% 见顶（234.7），随后持续下滑到 157.2；与此同时 contact_guidance 从 27.9 一路上涨到 331.9。**agent 在学会最大化 contact 奖励的同时，实际任务表现却在恶化。**

**iter_2 的教训（不要重蹈）：** potential-diff `pot_next - pot_prev` 把 proximity 和 velocity 也纳入了 potential，导致 >99% 的 step 都产生非零 delta（正负大幅振荡，signed_share 仅 2.8%）。signal 被噪声淹没。

### 2. Behavior Diagnosis（行为诊断）

当前策略学到了什么：**agent 发现停留在平台上可以持续收取 ~0.6/step 的 contact 奖励。** 从 agent 视角，settle（触发 episode 终止）会切断这条奖励流，因此它学会了"占着平台但不 settle"——尽可能延长在平台上的时间以最大化累计奖励。

这是不健康的。证据：
- terminated 仅 4/20，其余全部 truncation（864 步跑满）
- contact_guidance ep_sum 从 cp70% 的 28 涨到 cp100% 的 332，但 score 从 235 跌到 157
- agent 在平台上"磨蹭"而非快速稳定着陆

### 3. Signal Completeness（信号完备性）

| 职责 | 状态 | 问题 |
|---|---|---|
| goal_proximity（progress） | ✅ 有，弱 | 1.36/ep，提供接近引导但被 contact 淹没 |
| velocity_management | ✅ 有，极弱 | -0.18/ep，几乎不影响行为 |
| orientation_stabilization | ✅ 有，弱 | -1.1/ep，有效但量级小 |
| safe_contact | ❌ **被 exploit** | state-value 奖励"占据好状态"→ agent 拖延 settle |
| **缺：settling incentive** | ❌ | 当前无任何信号鼓励 agent 尽快完成着陆并 settle |

核心问题：**safe_contact 是 state-value（占据状态即获奖），缺少 settling incentive（完成着陆后应停止奖励）。**

### 4. Selected Level: **L2**

理由：state-value → transition-based 属于数学形态变更（"持续→状态转移事件"模式）。单纯调权重（L1）无法解决 exploit 的根本机制——只要 contact 是 state-value，agent 就会倾向于延长平台停留时间。

### 5. Selected Intervention

**目标组件：`contact_guidance`**

将 contact_guidance 从 **state-value（占据状态获奖）** 改为 **transition-based（仅奖励获得接触的转移事件）**：

```
旧：contact_guidance = 0.3 * contact_val * proximity_factor * velocity_ok
     → 每步在平台上可获奖 ~0.6，agent 拖延 settle

新：contact_guidance = 100.0 * max(0, contact_next - contact_prev) * proximity_factor
     → 仅在获得腿接触时获奖（每腿最多100），之后为零
     → agent 无动机拖延，接触后自然 settle
```

**与 iter_2 的关键区别：**
- iter_2 的 potential-diff 把 proximity + velocity 也纳入 potential，导致每步都非零（噪声淹没信号）
- 本轮只用 `max(0, contact_next - contact_prev)`，仅接触状态变化时非零，信号干净
- 不做负向惩罚（`max(0, ·)`），避免 iter_2 的大幅正负振荡

### 6. Falsifiable Hypothesis

如果 transition-based contact 正确，下一轮应观察到：
- contact_guidance 的 magnitude_share 显著下降（不再是 99.2%）
- contact_guidance active_rate 下降（不再每步触发，仅在接触瞬间触发）
- score 在 cp70% 之后不再下降（消除 exploitation）
- terminated 比例上升（agent 不再拖延，接触后更快 settle）
- 最终 score 向 200 靠近（两腿着陆 ≈ 100×2 + progress ~1.4 ≈ 201）

### 7. Expected Next Round

agent 行为模式：积极接近目标（progress 驱动）→ 到达平台附近后减速 → 双腿着陆获取 transition 奖励 → 无更多奖励可榨取 → 自然 settle → 成功终止。episode 长度应缩短，terminated 比例应上升。

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

    # --- contact guidance: transition-based (L2 change from state-value) ---
    # State-value contact reward (iter_1/iter_5) caused exploitation:
    # agent learns to occupy the platform collecting ~0.6/step indefinitely,
    # rather than settling to trigger successful termination.
    # Evidence: score peaked at cp70% (234.7) then declined to 157.2,
    # while contact_guidance rose from 27.9 to 331.9 over the same period.
    #
    # Transition-based: reward only the act of GAINING leg contact near target.
    # Once both legs are on, reward drops to zero — no incentive to linger.
    # Unlike iter_2's potential-diff (which included proximity/velocity in the
    # delta, creating 99.9% magnitude noise), this only fires on contact changes.
    contact_prev = float(obs[6] + obs[7])
    contact_next = float(next_obs[6] + next_obs[7])
    contact_gain = max(0.0, contact_next - contact_prev)   # only reward gaining legs
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)     # gate: must be near target
    contact_guidance = 100.0 * contact_gain * proximity_factor

    # --- combine all components ---
    # Progress weight=1.0 provides approach guidance.
    # Transition-based contact provides landing incentive (~200 for two legs).
    # Together they target ~200 score for a successful soft landing.
    total_reward = (1.0 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 1.0 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```