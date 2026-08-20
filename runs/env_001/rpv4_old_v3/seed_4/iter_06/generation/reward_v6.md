现在我已掌握完整图景。让我撰写诊断报告和代码。

---

## 1. 诊断报告

### evidence

| 证据来源 | 数据 |
|---|---|
| iter_5 score | -40.67（目标 200，差距 ~6x），range [-134.66, 7.00] |
| iter_5 terminated | **1/20**（仅 1 次成功着陆），truncated 19/20 |
| iter_5 len | 991.5（几乎满步，agent 在徘徊） |
| iter_5 contact active_rate | **0.1%（死组件）** — 比 iter_4 的 0.5% 更差 |
| iter_5 pos magnitude_share | **72.5%** — pos 主导所有信号 |
| checkpoint 40%→80% | score: -96.65 → -54.47（+42.18，训练在改善但基线极低） |
| checkpoint contact | 40%: 50.45 → 80%: 2.29（**训练过程中 contact 信号被主动放弃**） |
| 稳定性组件 5 轮演化 | vel/angle/angvel **5 轮完全不变**：gate=1/(1+10·dist)，w 均为 0.5 |
| 稳定性 magnitude_share 跨轮 | iter_1: 1.1% → iter_2: 0.8% → iter_3: 2.2% → iter_4: 13.1% → iter_5: 10.3% |
| 上一轮干预 (iter_5) | contact: binary→gradual (nleft+nright), threshold 0.3→0.5, w 10→6 |
| 上一轮干预结果 | contact active_rate 从 0.5% **进一步降到 0.1%**，score 从 +24.69 **暴跌到 -40.67** |
| best iter (iter_2) | score=64.24, contact magnitude_share=97.8%（sparse spike 驱动），terminated=10/20 |

### behavior_diagnosis

Agent 学会了「接近但不降落」的策略：pos 信号（距离改善量）提供了可靠的正奖励，agent 被驱动靠近着陆垫。但一旦接近，**没有任何有效信号迫使它减速、稳定、并完成两脚着陆**。稳定性惩罚（vel/angle/angvel）权重大小（w=0.5）且门控激活过晚（k_gate=10，在 dist=1.0 时 gate 仅 9%），agent 几乎感受不到减速压力。contact 组件在 5 轮中始终是死组件或接近死组件（active_rate ≤0.5% 除 iter_1），因为它过于稀疏，无法提供可学习的梯度。最终行为：agent 在着陆垫附近低速盘旋，直到 episode 被截断。

### signal_completeness

- `approach_progress` ✅ — pos 组件正常工作，驱动靠近
- `soft_landing` ❌ — 门控太晚、惩罚太弱，agent 感受不到减速/稳定压力
- `safe_contact` ❌ — 5 轮几乎全是死组件，稀疏信号无法学习
- `fuel_efficiency` ❌ — 从未引入，agent 可无成本连续开引擎

**缺失的关键信号**：有效的 soft_landing 引导。当前 gate 在 dist=0.5 时仅激活 16.7%，dist=1.0 时仅 9.1%——这意味着在 agent 最需要减速引导的中距离段（dist 0.3~1.5），稳定性惩罚几乎为零。

### selected_level

**L2** — 修改门控函数的数学行为（改 k_gate 参数改变激活曲线形状），配合稳定性权重上调。这不是纯系数调整（L1），因为改变 k_gate 会重塑 gate 在整个距离范围内的激活轮廓——从"仅近距激活"变为"中距开始渐进激活"。5 轮以来稳定性组件数学形态从未改变过，这是一个未探索的方向。

### selected_intervention

**目标组件：vel, angle, angvel（共享 gate）**

具体修改：
1. `k_gate`: 10.0 → **4.0**（在 dist=1.0 处 gate 从 9% 升至 20%，dist=0.5 处从 17% 升至 33%）
2. `w_vel`: 0.5 → **1.0**（加倍，使速度惩罚与 pos 可竞争）
3. `w_pos`: 8.0 → **6.0**（微降，防止 pos 淹没稳定性信号）
4. angle/angvel 权重和 contact 保持 iter_5 不变

这些修改服务于同一语义：**让稳定性引导在更大距离范围内、以更大强度激活**，迫使 agent 在中段就开始减速，而非仅在贴脸时才感受到微弱惩罚。

### falsifiable_hypothesis

如果 gate 提前激活且速度惩罚加倍，agent 将在中距离段（dist 0.3~1.5）感受到显著的减速压力。这会迫使它在中段降低速度，从而在接近着陆垫时有更可控的姿态和速度，提高 contact 触发概率。预期：vel penalty 的 magnitude_share 从 ~7% 升至 20-35%，contact active_rate 从 0.1% 回升至 2-10%，terminated 从 1/20 升至 3-8/20。

### expected_next_round

| 指标 | 当前 (iter_5) | 预期 (iter_6) | 理由 |
|---|---|---|---|
| score | -40.67 | -10 ~ +20 | 更强的减速引导应减少无效盘旋 |
| len | 991.5 | 700-900 | 更多 episode 以着陆而非截断结束 |
| terminated | 1/20 | 3-8/20 | 减速后着陆概率上升 |
| vel magnitude_share | 6.8% | 15-30% | w_vel 加倍 + gate 提前 → 更大贡献 |
| contact active_rate | 0.1% | 1-8% | 减速使精确着陆更可行 |
| pos magnitude_share | 72.5% | 40-55% | 被稳定性信号稀释 |

### main_risk

速度惩罚过强可能导致 agent 在远距离就过度减速，迟迟无法到达着陆垫（悬停恐惧）。如果下一轮出现 len 保持 1000 且 pos ep_sum 大幅下降（agent 不敢前进），则需回调 w_vel 或引入与距离解耦的双门控（近距强惩罚、远距零惩罚）。当前 k_gate=4.0 的选择是折中：在 dist=2.0 时 gate=11%，仍有微弱激活但不会过度抑制远距移动。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # next_obs indices:
    # 0: x_position (relative to pad center)
    # 1: y_position (relative to pad height)
    # 2: x_velocity
    # 3: y_velocity
    # 4: body_angle
    # 5: angular_velocity
    # 6: left_support_contact (0.0 or 1.0)
    # 7: right_support_contact (0.0 or 1.0)

    ox = obs[0]
    oy = obs[1]
    nx = next_obs[0]
    ny = next_obs[1]
    nvx = next_obs[2]
    nvy = next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft = next_obs[6]
    nright = next_obs[7]

    # Distances
    prev_dist = (ox**2 + oy**2 + 1e-8) ** 0.5
    next_dist = (nx**2 + ny**2 + 1e-8) ** 0.5

    # -- 1) approach progress: reward reduction in distance to pad --
    w_pos = 6.0
    pos_reward = w_pos * (prev_dist - next_dist)   # >0 when approaching, <0 when retreating

    # -- 2) soft landing: speed & attitude penalty gated by distance --
    # Gate activates earlier: at dist=1.0, gate=0.20 (was 0.09 with k=10);
    # at dist=0.5, gate=0.33 (was 0.17); at dist=0.2, gate=0.56 (was 0.33).
    k_gate = 4.0
    gate = 1.0 / (1.0 + k_gate * next_dist)

    w_vel = 1.0
    vel_penalty = -w_vel * (nvx**2 + nvy**2) * gate

    w_angle = 0.5
    angle_penalty = -w_angle * (nangle**2) * gate

    w_angvel = 0.5
    angvel_penalty = -w_angvel * (nangvel**2) * gate

    # -- 3) safe contact: reward each leg touching near center (gradual) --
    contact_proxy = nleft + nright                    # 0, 1, or 2
    contact_threshold = 0.5
    contact_factor = max(0.0, 1.0 - next_dist / contact_threshold)
    w_contact = 6.0
    contact_reward = w_contact * contact_proxy * contact_factor

    total_reward = pos_reward + vel_penalty + angle_penalty + angvel_penalty + contact_reward

    components = {
        'pos': pos_reward,
        'vel': vel_penalty,
        'angle': angle_penalty,
        'angvel': angvel_penalty,
        'contact': contact_reward
    }

    return total_reward, components
```