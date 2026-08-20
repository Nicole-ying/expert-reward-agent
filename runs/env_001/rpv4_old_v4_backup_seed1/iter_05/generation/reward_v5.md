现在我已经掌握了全部数据。让我进行系统诊断。

---

## 1. 诊断报告

### evidence

**跨轮核心数据对比：**

| iter | score | terminated | goal_proximity 形态 | contact_bonus | 关键变化 |
|------|-------|-----------|---------------------|---------------|---------|
| 1 | -76.56 | 1/20 | 线性 Δd, w=1.0 | 50 (dead) | 基线 |
| 2 | **63.72** | **17/20** | 线性 Δd, w=2.0 | 50 (5.2%, dominates 98.2%) | vel 加高度门控 |
| 3 | 21.45 | 9/20 | 线性 Δd, w=2.0 | 50+速度门控 (8.2%, 97.7%) | contact 加速度门控 |
| 4 | -10.11 | 0/20 | **逆距离 Δ**, w=3.0 | 8 (dead) | goal 改逆距离, contact 减至 8 |

**iter 4 checkpoint 崩溃证据：**
- cp90%: score=**+59.6**, contact_bonus=158.68 (active 2.5%), goal_proximity=3.26
- cp100%: score=**-8.6**, contact_bonus=0.00 (dead), goal_proximity=3.06

在 cp90%→cp100% 之间，策略从能着陆彻底崩溃为无法着陆。这是典型的训练不稳定——逆距离势能在接近目标时梯度爆炸，导致策略在最后阶段被"甩出"已学到的良好区域。

**逆距离势能的梯度问题：**
- 在 d=10（典型起始距离），梯度 ∝ 1/(10+0.5)² ≈ 0.009
- 在 d=1，梯度 ∝ 1/(1+0.5)² ≈ 0.44
- 在 d=0.1，梯度 ∝ 1/(0.1+0.5)² ≈ 2.78

远处梯度比近处弱 **~300 倍**。agent 在大部分航程中几乎收不到方向信号，只有靠近目标时才突然获得巨大梯度——这解释了 cp90% 能着陆但 cp100% 崩溃的现象。

**语义振荡分析：**
- contact_bonus 已修改 3 轮（iter 3 加门控 → iter 4 去门控+降幅），属于反复振荡，不应再动
- goal_proximity 仅修改 1 次（iter 4 线性→逆距离），且这次修改直接导致崩溃
- velocity_penalty 自 iter 2 以来稳定有效

### behavior_diagnosis

当前策略的失败行为：**agent 完全不降落，全程漂移/悬浮直到 1000 步截断**。0/20 终止，contact_bonus 完全死亡。这是因为逆距离势能在中远距离（d>2）提供的梯度几乎为零，agent 收不到任何"朝目标移动"的有效信号。两个惩罚组件（velocity_penalty -7.84, upright_penalty -5.11）主导了 reward，agent 学到的最优策略就是尽量不动以最小化惩罚——这正好是悬浮行为。

### signal_completeness

必要职责完备性检查：
- `goal_proximity` ✓ 存在，但形态错误导致「有组件无信号」
- `soft_landing_velocity_penalty` ✓ 高度门控，合理
- `upright_orientation` ✓ 合理
- `contact_success` ✗ 存在但死亡——因为 agent 根本到不了着陆阶段

根本缺陷：`goal_proximity` 的数学形态（逆距离 delta）无法在全程提供有效梯度。

### selected_level

**L2** — 修改 goal_proximity 的数学形态（逆距离 delta → 线性距离 delta），这是形态级修改，不仅调系数。

### selected_intervention

**目标组件：`goal_proximity`**

将逆距离势能 delta 恢复为线性距离 delta，权重设为 4.0（高于 iter 2 的 2.0，因为当前 contact_bonus=8 远小于 iter 2 的 50，需要更强的连续引导信号来弥补）。

其他组件（velocity_penalty、upright_penalty、contact_bonus）保持不变。

### falsifiable_hypothesis

线性距离 delta `prev_dist - next_dist` 在任意距离下提供恒定的单位梯度（每接近 1 单位距离 = +4.0 奖励），与距离远近无关。这将：
1. 给 agent 从起点到终点的**全程均匀方向信号**，消除"远处无梯度"问题
2. 使 goal_proximity 的 episode 总量足以对抗两个惩罚组件（~+40 vs ~-13），净连续信号为正
3. agent 能被引导到目标附近，从而有机会触发 contact_bonus 并学习着陆

### expected_next_round

- **score**：应从 -10 回升到正区间（目标 > 0，理想 > 20）
- **terminated**：应从 0/20 恢复到 > 5/20
- **len**：应从 1000 下降到 < 900
- **goal_proximity ep_sum**：应显著增加（因为每步 delta 更大，且在更少步数内走完更远距离）
- **contact_bonus active_rate**：应从 0% 回升（agent 能到达目标附近后自然会触发）
- **Checkpoint 趋势**：不应再出现 cp90%→cp100% 的崩溃

### main_risk

线性 delta 在远距离时可能奖励过快接近（agent 可能高速冲向目标），velocity_penalty 的高度门控需要在近地处提供足够强的减速信号来平衡。权重 4.0 若过高，可能让 agent 忽视减速。若下一轮出现高速撞击，需要微调 goal_proximity 权重或收紧 velocity_penalty。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 5): goal_proximity reverted from inverse-distance delta
    back to linear distance delta. Inverse-distance potential (iter 4) caused
    catastrophic training collapse: negligible gradients at moderate distances
    meant the agent received almost no approach signal, learning to hover
    instead. Linear delta provides uniform per-step gradient at all distances,
    which was proven to work in iter 2 (score 63.72, 17/20 terminated).
    Weight raised to 4.0 (from iter 2's 2.0) to compensate for the smaller
    contact_bonus (8 vs 50) and ensure net continuous signals are positive.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: moderate reward for dual-leg contact ---
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian decay with horizontal offset from pad center
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 8.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```