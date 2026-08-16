# Response Record

现在我对整个演化历史有了完整的理解。让我来撰写诊断报告。

---

## 诊断报告

### Evidence（关键证据）

**历史演化轨迹（iter_1 → iter_4）：**

| iter | contact 公式 | progress 权重 | score | len | terminated | contact ep_sum | contact active% | progress signed_share |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| 1 | state-value (0.3) | **0.5** | **171.4** | 781 | 7/20 | 252.9 | 68.0% | 0.3% |
| 2 | potential-diff (100) | 2.0 | 157.1 | 926 | 3/20 | 155.5 | 81.0% | ~0% |
| 3 | potential-diff (3.0) | 5.0 | -73.6 | 96 | 20/20 | 3.8 | 0.9% | 52.8% |
| 4 | state-value (0.3) | **5.0** | **-52.5** | 72 | 20/20 | 0.25 | 0.8% | **82.2%** |

**iter_4 checkpoint 趋势揭示的僵局：**
- Score 全程为负（-45 到 -99），从未转正
- contact_guidance **全程死亡**：ep_sum 始终在 0.20–0.32 之间平躺，active_rate 始终 0.7%–1.0%
- progress 全程平坦（5.53–5.59），说明 agent 在训练早期就已收敛到单一策略，之后无任何行为改变
- orientation_penalty 和 velocity_penalty 也全程平坦

**对比 iter_1 的成功轨迹（从 iter_2 reflection）：**
- iter_1 在 cp70% 出现「发现时刻」：contact_guidance 从 0 骤升至 248.8，score 从 -15.9 跃升至 105.8
- iter_4 从未出现这样的发现时刻 — 全训练过程 contact_guidance 始终为 0

### Behavior Diagnosis

当前策略学到的行为模式：**全速冲向原点（目标），撞击地面后 crash**。

证据：
- **len=72**（极短），20/20 全部 terminated（crash），无一存活
- progress signed_share=82.2%，magnitude_share=85.7% — agent 唯一在乎的就是「接近目标」
- contact_guidance active_rate=0.8% — agent 几乎从未经历过接触奖励，根本没机会发现「接触=好」
- velocity_penalty=-0.47 很小 — agent 没有减速动机

这不是健康的策略。Agent 被单一的 progress 信号完全支配，形成了「以最快速度撞向目标」的局部最优。因为 episode 太短（~72步），agent 没有足够的探索步数去偶然触发 contact_guidance 并发现其价值。

### Signal Completeness

必要职责检查：
- ✅ `goal_proximity`：progress 信号存在且工作（agent 确实在接近目标）
- ❌ `velocity_management`：velocity_penalty 太弱（ep_sum=-0.47），无法对抗 progress 的拉力
- ✅ `orientation_stabilization`：存在但量级小
- ❌ `safe_contact`：**contact_guidance 实际上已死亡**。公式正确（与 iter_1 相同），但 progress 太强导致 agent 在触发 contact 之前就已 crash

缺失的信号：contact_guidance 需要被 agent 实际经历到，才能进入 RL 的奖励信号。当前它被 progress 完全淹没。

### Selected Level

**L1** — 只调系数。

理由：
1. iter_1 用相同的数学公式（state-value contact_guidance + progress + velocity_penalty + orientation_penalty）达到了 171 分
2. iter_4 与 iter_1 的唯一实质性差异是 progress 权重：**5.0 vs 0.5，差了 10 倍**
3. 问题不是公式错了，而是权重比严重失衡。这是纯 L1 问题
4. best score (171) 距目标 (200) 差 14.3%，已接近目标，L1 微调优先

### Selected Intervention

**目标组件**：`progress`

**具体修改**：`5.0 * progress` → `1.0 * progress`

保持 contact_guidance=0.3、velocity_penalty、orientation_penalty 完全不变。

### Falsifiable Hypothesis

**因果链**：progress 权重从 5.0 降到 1.0 →
- Agent 不再被极端强烈的「冲向目标」信号支配
- Episode 变长（预计 200–500 步），agent 有更多探索步数
- 在更长的 episode 中，agent 有更高概率偶然满足 `contact_val>0 AND proximity_factor>0 AND velocity_ok>0` 三个条件
- 一旦 contact_guidance 被触发，其单步量级（最大 0.6）远大于 progress 单步量级（~0.01–0.02），agent 将学会在接近目标后减速并寻求接触
- 类似于 iter_1 的「发现时刻」将再次出现

**可证伪条件**：如果下一轮 contact_guidance active_rate 仍然 < 5%，说明权重调整不足以解决发现问题，需要考虑 L2（如将 contact_guidance 的三因子乘积改为加性结构以降低稀疏性）。

### Expected Next Round

- score：预期回升到 50–150 区间（不保证回到 171，因为 1.0 是 iter_1 的 0.5 的两倍，可能仍需微调）
- len：预期 200–600 步
- terminated：预期下降到 10–16/20（部分 episode 能成功着陆）
- contact_guidance active_rate：预期从 0.8% 上升到 10–40%
- 训练过程中应出现 contact_guidance 的「发现时刻」

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

    # --- contact guidance: state-based occupancy reward ---
    # Same formula as iter_1 (score=171): reward both legs touching,
    # but only near the target and at low speed.
    contact_val = float(next_obs[6] + next_obs[7])   # range 0.0 .. 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    velocity_ok = max(0.0, 1.0 - abs(vy) / 0.5)
    contact_guidance = 0.3 * contact_val * proximity_factor * velocity_ok

    # --- combine all components ---
    # Progress weight reduced from 5.0 to 1.0 (L1).
    # At 5.0, progress dominated (85.7% magnitude_share), causing the agent
    # to rush toward the target and crash before ever discovering contact.
    # At 0.5 (iter_1), contact_guidance dominated once discovered (99.2% share).
    # 1.0 is a middle ground: stronger approach than iter_1 but still allows
    # contact discovery (contact_guidance max ~0.6/step >> progress ~0.01-0.02/step).
    total_reward = (1.0 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 1.0 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```
