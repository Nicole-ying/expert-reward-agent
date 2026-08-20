Now I have the full picture. Let me synthesize everything and write the diagnosis.

---

## 1. 诊断报告

### evidence

**跨轮核心数据（7 轮迭代全景）：**

| iter | score | term | contact_bonus | ep_sum | active | mag_share | vel_penalty 权重 | vel gate |
|------|-------|------|---------------|--------|--------|-----------|-----------------|----------|
| 1 | -76.56 | 1/20 | 50 | 0 | 0% | 0% | -0.5 | 无 |
| 2 | **63.72** | **17/20** | 50 | 1075.63 | 5.2% | 98.2% | -0.5 | exp(-h/2) |
| 3 | 21.45 | 9/20 | 50+速度门控 | 1011.47 | 8.2% | 97.7% | -0.5 | exp(-h/2) |
| 4 | -10.11 | 0/20 | 8 | 0 | 0% | 0% | -0.5 | exp(-h/2) |
| 5 | -33.04 | 0/20 | 8 | 0 | 0% | 0% | -0.5 | exp(-h/2) |
| 6 | -81.80 | 0/20 | landing_approach(连续) | 0.974 | 98.3% | — | **-0.3** | exp(-h/2) |
| 7 | -5.37 | 2/20 | 20 | 95.90 | 0.7% | 86.1% | **-0.3** | exp(-h/2) |

**本轮 (iter 7) 关键数据：**
- 得分 -5.37，仅 2/20 成功着陆，18/20 超时截断（len=964）
- contact_bonus magnitude_share=86.1%，但 active_rate 仅 0.7%——稀疏大额奖励主导总奖励，但只在极少数时刻触发
- goal_proximity ep_sum=4.754（100% active），velocity_penalty=-4.621（100% active），upright_penalty=-5.455（100% active）
- 三个连续信号合计约 -5.32/episode，几乎被 contact_bonus 的 +95.9 淹没（但当 contact_bonus 不触发时，agent 只有微弱的连续信号）

**Checkpoint 轨迹揭示的不稳定性：**
- cp30%: contact_bonus=**2054.686**（active=25.2%），说明 agent 在训练中期曾大量成功着陆
- cp40%→cp100%: contact_bonus 崩溃至 4.608，着陆能力完全退化
- 得分从 cp30%=118.6 → cp60%=-67.4 → cp80%=111.8 → cp100%=-13.5，剧烈振荡

**语义演变追踪（关键发现）：**
- `contact_bonus` 语义已修改 6 次（v1→v7 经历 50→50+gate→8→8→移除→20），2 次稀疏↔连续切换，1 次速度门控添加→移除。**该语义已穷尽，不应再动。**
- `velocity_penalty` 语义：权重从 -0.5→-0.3（iter 6），但 **gate 衰减常数 `exp(-h/2.0)` 从 iter 2 以来从未改变**。该语义还有未探索的参数空间。
- `goal_proximity` 语义：线性↔逆距离切换 1 次，权重从 1→4 单调递增。线性形式在 iter 2 验证有效。
- `upright_penalty` 语义：**7 轮完全未修改**，始终 `-1.0*(angle²+angvel²)`。
- `fuel_efficiency`：从未引入。

### behavior_diagnosis

**agent 当前失败行为：接近但不着陆。**

- agent 学会了向目标垫移动（goal_proximity=+4.75，100% active），但无法可靠完成从"接近"到"着陆"的最后一段
- velocity_penalty 仅 -4.62/episode（per-step ~0.0048），对速度的约束太弱——agent 可以在近地区域保持较高速度而不受足够惩罚
- 对比 iter 2（最佳，17/20 成功）：velocity_penalty=-9.52（per-step ~0.014），几乎 3 倍于当前约束力
- 当前 `exp(-h/2.0)` gate 衰减太慢：在 height=1.0 时 gate=0.607，在 height=0.3 时 gate=0.861——门控几乎没有起到"近地集中惩罚"的作用
- 训练不稳定：agent 偶尔发现着陆（cp30%），但策略无法巩固——因为从"快速下降"到"软着陆"之间缺乏足够陡峭的梯度信号来塑造减速行为

**根因：velocity_penalty 既不够强（权重 0.3 vs 历史上有效的 0.5），高度门控也不够锐利（gate 在近地和远地差异太小），导致近地减速梯度不足以引导可靠着陆。**

### signal_completeness

环境卡片定义的 4 个 mandatory_roles 均已覆盖：
- `goal_proximity` ✓（线性 Δd，权重 4.0）
- `soft_landing_velocity_penalty` ✓（但强度/锐度不足）
- `upright_orientation` ✓（权重 -1.0，7 轮未变）
- `contact_success` ✓（稀疏 bonus=20，主导但不稳定）

缺失的不是职责覆盖，而是 `soft_landing_velocity_penalty` 的**近地梯度质量**——当前门控无法区分"离地 5m"和"离地 0.5m"的惩罚紧迫性差异。

### selected_level

**L2** — 修改 velocity_penalty 的数学形态（gate 衰减常数）+ 权重调整。虽然 gate 公式保持 `exp(-h/k)`，但改变 k 从 2.0→1.0 并配合权重 0.3→0.5，改变了整个组件的行为剖面：从"均匀弱约束"转为"近地锐利约束 + 高空自由"。这不是简单的参数微调，而是行为模式的质变。

### selected_intervention

**目标组件：`velocity_penalty`**

修改内容：
1. **Gate 锐化**：`exp(-height / 2.0)` → `exp(-height / 1.0)`。衰减加快一倍，使惩罚更集中在地面附近。
2. **权重回升**：`-0.3` → `-0.5`。恢复 iter 2 验证有效的约束强度。

效果对比（以速度 v²=1 为例）：

| 高度 | 当前 (w=0.3, k=2.0) | 新 (w=0.5, k=1.0) | 变化 |
|------|---------------------|-------------------|------|
| h=3.0 | 0.067 | 0.025 | **-63%**（高空更自由） |
| h=1.0 | 0.182 | 0.184 | ≈0%（过渡点） |
| h=0.5 | 0.234 | 0.303 | **+30%**（近地更强） |
| h=0.0 | 0.300 | 0.500 | **+67%**（地面最强） |

这直接回应了 iter 6 "penalty wall 阻碍接近"的顾虑——新 gate 在高空反而更弱，agent 可以自由移动接近目标；但一旦进入近地区域（h<1），惩罚快速攀升，迫使减速。

### falsifiable_hypothesis

**如果 hypothesis 正确**：更锐利的近地速度惩罚将迫使 agent 在接近地面时主动减速，减速后的低速状态自然增加双腿接触的概率，从而提高着陆成功率。预期 terminated 从 2/20 提升到 ≥8/20，contact_bonus ep_sum 应上升（更多着陆事件），velocity_penalty ep_sum 应上升（绝对值增大，因为近地惩罚更强），训练稳定性改善（checkpoint 得分振荡减小）。

**如果 hypothesis 错误**：agent 会在近地惩罚墙前停滞（悬停），terminated 保持低位但 len 缩短（agent 学会悬停在高处避免惩罚），或 velocity_penalty magnitude_share 急剧上升淹没其他信号。

### expected_next_round

- **score**：预期回升至 0~40 区间（从 -5.37 改善）
- **terminated**：预期 ≥5/20（从 2/20 提升）
- **velocity_penalty ep_sum**：预期绝对值增大至 -8~-12（从 -4.62），因为近地惩罚更强
- **velocity_penalty magnitude_share**：预期上升至 15-30%（从 4.1%），因为加强了权重
- **contact_bonus ep_sum**：预期上升（更多着陆）
- **goal_proximity**：预期保持正值，不应显著退化
- **训练稳定性**：预期 checkpoint 轨迹的振荡幅度减小

### main_risk

**近地惩罚墙导致悬停**：如果 gate 太锐利（k=1.0 在 h=0.2 时 gate=0.819，已经很陡），agent 可能学会在 ~1m 高度悬停以避免触发强惩罚，而不是穿过惩罚区完成着陆。缓解因素：contact_bonus=20 提供了穿越惩罚区的正向激励，且 k=1.0 的过渡仍然平滑（指数函数连续可微），不会产生悬崖式梯度断裂。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 8): velocity penalty gate sharpened from exp(-h/2.0)
    to exp(-h/1.0) and weight raised from 0.3 to 0.5. This concentrates
    the soft-landing penalty near the ground (h < 1.0) while giving the
    agent more freedom at altitude. The sharper near-ground gradient forces
    deceleration in the final descent phase, bridging the gap between
    "approaching the pad" and "actually landing."

    Evidence: iter 7 had only 2/20 landings despite active goal_proximity.
    The velocity penalty at -0.3 with the wide gate (exp(-h/2)) was too
    diffuse to create a clear "slow down now" signal near the surface.
    Iter 2 (best, 17/20) used w=0.5, and the sharper gate retains altitude
    freedom while making the near-ground braking signal even stronger than
    iter 2's configuration.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: sharply height-gated ---
    # gate = 1.0 at ground, decays rapidly with height (k=1.0 vs old k=2.0)
    # At h=1.0: gate=0.368 (old: 0.607) — weaker, more freedom at altitude
    # At h=0.3: gate=0.741 (old: 0.861) — comparable
    # At h=0.0: gate=1.000 — full penalty strength
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 1.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: moderate terminal reward for dual-leg landing ---
    # Value 20 kept from iter 7 — large enough to incentivize landing,
    # small enough not to dominate when continuous signals are healthy.
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 20.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return float(total_reward), components
```