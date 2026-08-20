# Response Record

现在我已掌握全貌。以下是诊断和修改。

---

## 1. 诊断报告

### evidence
- **Iter 1→2 变化回顾**：上一次干预是将 velocity_penalty 从全高度惩罚改为高度门控（`exp(-height/2.0)`），并将 goal_proximity 权重从 1.0→2.0。效果显著：得分从 -76.56 跃升至 63.72，terminated 从 1/20 → 17/20，contact_bonus 从完全死亡（0%）激活到 5.2%。
- **当前 contact_bonus 主导**：magnitude_share=98.2%（>90% 阈值），ep_sum=1075.63。每步均值 1.61，而其他三个连续信号合计仅 ~0.028。contact_bonus 在量级上压倒一切。
- **Checkpoint 异常**：cp70% 时 contact_bonus=4038.918，active_rate=16.0%——远高于 cp100% 的 746.69 / 3.7%。这表明训练中期 agent 曾大量反复触发 contact（可能着陆后弹跳重复接触），后期有所收敛但仍偏高。
- **连续信号过于微弱**：velocity_penalty 每步仅 -0.014，upright_penalty -0.011，goal_proximity +0.003。agent 几乎完全由稀疏的 contact_bonus 驱动，缺乏中间行为（减速、姿态控制）的有效梯度。
- **得分仍有差距**：63.72 vs 目标 200，range=[-208, 266] 说明个别 episode 仍严重失败。

### behavior_diagnosis
Agent 学会了到达目标垫并触发双腿接触（17/20 成功终止），但**着陆质量不高**——可能在接触时仍有较高速度。contact_bonus 对高速接触和低速软着陆一视同仁，agent 没有动力在触地前精细减速。同时 contact_bonus 的稀疏巨额奖励淹没了所有连续引导信号，使速度控制和姿态稳定缺乏有效学习梯度。

### signal_completeness
四个必要职责（goal_proximity、soft_landing_velocity_penalty、upright_orientation、contact_success）均已存在，不缺信号。问题是**量级失衡**而非信号缺失。

### selected_level
**L2** — 修改 contact_bonus 的数学形态：在原有的 centering_factor 基础上追加速度质量门控。服务于"软着陆质量"这一语义。

### selected_intervention
**目标组件：contact_bonus**

在现有 `centering_factor` 基础上追加 **velocity_quality gate**：`exp(-3.0 * speed_sq)`，其中 `speed_sq = vx² + vy²`。只有接近零速度的软着陆才能获得接近满额的 50.0 bonus；高速撞击时 bonus 趋近于零。这会迫使 agent 在触地前主动减速。

### falsifiable_hypothesis
速度门控将使 contact_bonus 的 magnitude_share 从 98% 下降到更健康的水平（预期 60-80%），同时 velocity_penalty 的相对贡献上升。Agent 将被迫在接近地面时降低速度以获取 contact_bonus，从而改善着陆质量，提升得分。

### expected_next_round
- contact_bonus：magnitude_share 下降（高速接触不再获得大额 bonus），ep_sum 可能先降后升
- velocity_penalty：magnitude_share 上升（相对权重增大），但绝对值可能缩小（agent 减速更好）
- 得分：可能初期短暂下降（bonus 变难获取），但 checkpoint 中后期应超过当前 63.72
- terminated 率：可能暂时略降后回升

### main_risk
速度门控过严可能导致 agent 在探索阶段完全拿不到 contact_bonus，失去"到达目标垫"的动力。`exp(-3.0)` 的衰减系数中等偏温和——在 speed=1.0 时 quality≈0.05，speed=0.5 时 quality≈0.22，仍保留了一定梯度。若下一轮 contact_bonus 再次死亡，需要放宽系数。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 3): contact_bonus now velocity-gated via
    exp(-3.0 * speed_sq). Only low-velocity contacts receive the full bonus;
    high-speed touchdowns are strongly penalized by near-zero bonus.
    This forces the agent to decelerate before ground contact rather than
    relying on the sparse bonus to mask hard landings.
    """
    # --- goal proximity: encourage reducing Euclidean distance to pad ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 2.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: velocity-gated soft landing reward ---
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian decay with horizontal offset from pad center
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        # Velocity quality gate: only reward near-zero-speed contacts
        speed_sq = next_obs[2]**2 + next_obs[3]**2
        velocity_quality = 2.718281828 ** (-3.0 * speed_sq)
        contact_bonus = 50.0 * centering_factor * velocity_quality

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```
