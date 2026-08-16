现在我已掌握全局。让我整理诊断报告并给出修改方案。

---

## 1. 诊断报告

### evidence

- **iter_1**: score=-159.49, 20T/0Tr, len=303。`contact_reward` 死亡（0%），`vel_damping` 死亡（0%）。仅靠 `progress_reward`（+0.10/ep）和 `orientation_penalty`（-0.03/ep）即实现 100% 着陆。说明**基础引导信号足以让 agent 到达目标垫**。
- **iter_2**: 我引入状态高斯 `landing_proximity_reward`（σ_dist=2.0, w=3.0），ep_sum=2704，magnitude_share=99%。Agent 学会在距离 ≈2.0 处盘旋收集奖励，**零着陆**（0T/20Tr, len=1000）。exploit 根源：σ_dist=2.0 太宽，中等距离处仍有显著奖励。
- **iter_3**: 我将其改为 progress-weighted（approach × speed × angle），意图消除盘旋 exploit。结果：`landing_proximity_reward` ep_sum 从 2704 暴跌至 25.5（-99%），`progress_reward` 变为负值（-1.975），checkpoint 显示 score 在 cp50%=-211.8 达到峰值后持续恶化至 cp100%=-514.5。**新 exploit：振荡套利**——agent 学会后退再前进以反复触发 approach 奖励。
- **vel_damping**：三轮全部死亡或接近死亡（dist_thresh=1.0 从未被足够接近以触发）。
- **soft_landing_success**（环境卡片标注的 mandatory role）：三轮从未实现。

### behavior_diagnosis

策略当前处于**信号真空**：`landing_proximity_reward` 因 progress-weighting 变得过弱且创造振荡 incentive，agent 在"靠近-后退-靠近"循环中消耗时间（len=967），几乎不着陆（1/20 terminated）。`progress_reward` 为负值证实了振荡行为。

### signal_completeness

- `goal_proximity`（mandatory）：`progress_reward` 提供 delta-distance，但信号弱（±1/step）且当前被振荡行为支配。
- `soft_landing_success`（mandatory）：**完全缺失**。没有任何组件在成功着陆时给予正向奖励。
- `safe_orientation`：`orientation_penalty` 正常工作，但 active_rate=100% 且 magnitude_share=2.1%，被淹没。
- `velocity_damping_near_pad`（conditional）：`vel_damping` 死亡，因为 dist_thresh=1.0 从未激活。

**核心缺口**：着陆/接近信号在 iter_2（太容易被 exploit）和 iter_3（太弱+振荡 exploit）之间摇摆，从未找到正确形态。

### selected_level

**L2** — 改变 `landing_proximity_reward` 的数学形态。这不是简单的系数调整（L1），因为 progress-weighted 形态本身有振荡套利的根本缺陷。也不是 L3，因为不需要换框架——状态高斯的基本思路（iter_2）方向正确，只是 σ 太宽。

### selected_intervention

**将 `landing_proximity_reward` 从 progress-weighted 形态改回状态高斯，但使用更紧的 σ（σ_dist=0.7 vs iter_2 的 2.0），并提高权重以补偿窄高斯。**

具体：`landing_proximity_reward = w_landing × exp(-dist²/σ_dist²) × exp(-speed²/σ_speed²) × exp(-angle²/σ_angle²)`

- `σ_dist = 0.7`（iter_2 为 2.0，收紧 2.86×）
- `σ_speed = 0.8`（iter_2 为 1.0）
- `σ_angle = 0.3`（不变）
- `w_landing = 10.0`（iter_2 为 3.0，iter_3 为 5.0）

数值验证：
- dist=2.0（iter_2 exploit 区）：exp(-4/0.49) = 0.00028，reward ≈ 0.003/step —— **近乎为零，无法 exploit**
- dist=0.5（近垫）：exp(-0.25/0.49) = 0.60，reward ≈ 6.0/step —— **有意义的正向信号**
- dist=0（垫上）：reward = 10.0/step —— **峰值明确**

### falsifiable_hypothesis

紧 σ 状态高斯消除了 iter_2 的"中等距离盘旋"exploit（σ_dist=0.7 使 dist>1.5 区域奖励归零），同时避免了 iter_3 的振荡 exploit（状态奖励是单调的：越近越好，无后退 incentive）。`progress_reward` 提供全局接近梯度，`landing_proximity_reward` 在近垫区提供强正向信号引导最终着陆。两者协同应使 agent 恢复着陆能力。

### expected_next_round

- `landing_proximity_reward` active_rate：预计 30-60%（仅在近垫区激活，不像 iter_2 的 100%）
- `progress_reward` ep_sum：应回到正值（不再振荡套利）
- `terminated` 比例：应恢复到至少 iter_1 水平（≥10/20）
- `score`：预期 -200 ~ -100（比 iter_3 大幅改善，但若无成功奖励可能仍为负）
- `vel_damping`：可能开始激活（agent 被激励进入 dist<1.0 区域）

### main_risk

**近垫盘旋**：如果环境 termination 阈值非常严格（如需要 dist<0.05），agent 可能在 dist≈0.2-0.5 处获得显著奖励而不触发 termination。缓解因素：iter_1 中 agent 在 303 步内自然终止 20/20，说明 termination 阈值在实践中是可达到的。如果此风险在下轮实现，下一步应添加 contact-based 成功奖励（稀疏但大幅）。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v4: landing_proximity_reward changed back to state-Gaussian but with tight sigma
        (sigma_dist=0.7 vs iter_2's 2.0) to eliminate the hovering-at-moderate-distance
        exploit while avoiding the iter_3 oscillation exploit from progress-weighting.
        The tight Gaussian is near-zero beyond dist~1.5, so the agent must genuinely
        approach the pad to receive meaningful reward.
    """
    # Extract relevant signals
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # 1. Progress reward (improvement_delta on Euclidean distance)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # positive when approaching
    w_progress = 1.0
    progress_reward = w_progress * progress

    # 2. Orientation penalty (quadratic_penalty)
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Landing proximity reward — tight state-Gaussian (NOT progress-weighted)
    #    sigma_dist=0.7 ensures reward is near-zero beyond ~1.5 units, eliminating
    #    the iter_2 exploit where the agent hovered at moderate distance.
    #    sigma_speed=0.8 and sigma_angle=0.3 gate the reward behind soft-landing
    #    conditions (low speed, upright), which naturally occur during final approach.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    sigma_dist = 0.7
    sigma_speed = 0.8
    sigma_angle = 0.3

    dist_factor = 2.718281828 ** (- (dist_curr ** 2) / (sigma_dist ** 2))
    speed_factor = 2.718281828 ** (- (speed ** 2) / (sigma_speed ** 2))
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))

    w_landing = 10.0
    landing_proximity_reward = w_landing * dist_factor * speed_factor * angle_factor

    # 4. Near-pad velocity damping (hinge / gated penalty)
    dist_thresh = 1.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 at far, 1 at origin
    w_vel_damp = 0.5
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    total_reward = progress_reward + orientation_penalty + landing_proximity_reward + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_proximity_reward": landing_proximity_reward,
        "vel_damping": vel_damping
    }
    return float(total_reward), components
```