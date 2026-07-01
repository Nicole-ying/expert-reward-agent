# Response Record

Let me analyze the situation carefully.

**Diagnosis:**

1. **score = -114.99**, all episodes terminate early (100% early terminal). The agent is crashing before landing.
2. **progress_delta_reward** is working well (nonzero_rate=100%, ratio=1.0) - agent is moving toward target.
3. **soft_landing_proxy** has nonzero_rate=0.72% - still extremely sparse! The continuous product form I introduced didn't help enough.
4. **stability_penalty** ratio=-0.41 - reasonable now, not dominating.
5. **original_env_reward** = -1.62 mean, -100 ratio - the environment itself is giving very negative rewards, meaning the agent is crashing.

The core problem: **Agent moves toward target but crashes before completing a soft landing.** The progress_delta reward encourages getting closer, but there's no strong enough signal to actually land softly. The soft_landing_proxy is still too sparse (0.72% nonzero rate).

Let me think about what to do:

The soft_landing_proxy has 4 factors multiplied together: dist_factor * speed_factor * angle_factor * contact_factor. Even though each factor is continuous, the product of 4 factors means it's near zero unless ALL conditions are met simultaneously. The contact_factor (next_obs[6] + next_obs[7])/2 might be the bottleneck - both legs need to be in contact.

Let me look at the data more carefully. The episode length is 74.1 steps. The agent is crashing. The progress_delta is positive (0.016) meaning it's moving toward target. But it never achieves a soft landing.

I think the issue is:
1. The soft_landing_proxy is still too hard to achieve - the product of 4 continuous factors means it's near zero unless all are good.
2. The agent needs a stronger shaping signal to slow down and stabilize as it approaches.

Let me try a **potential_based_shaping** approach which is theoretically guaranteed to preserve optimal policy, and can combine multiple objectives (distance + speed + angle) into one coherent signal.

Actually, let me think more carefully. The key insight is:

- progress_delta_reward works (ratio=1.0) - agent moves toward target
- But agent crashes (score=-115, all early terminal)
- soft_landing_proxy is too sparse (0.72%)

The problem is that the agent is incentivized to move toward the target quickly (progress_delta), but not incentivized enough to slow down and stabilize when it gets there. The stability_penalty is distance-gated so it's weak far away, but even near the target it might not be strong enough.

Let me try a different approach: **Replace progress_delta_reward with potential_based_shaping** that includes distance, speed, and angle. This way the agent is guided to simultaneously approach AND slow down AND stabilize.

Actually, let me also reconsider the soft_landing_proxy. The product of 4 factors is still too sparse. Let me make it a sum of factors instead, or use a different form.

Let me try a hybrid approach:
1. Keep progress_delta but reduce its weight
2. Add a bounded_proximity_reward as a dense signal for being near target
3. Make soft_landing_proxy a sum (not product) of factors to increase nonzero rate
4. Keep stability_penalty but make it stronger near target

Actually, let me try the potential_based_shaping approach. It's elegant and theoretically sound.

Potential function: Φ = -(distance + 0.5*speed + 0.3*abs(angle))
Then: shaping = γ * Φ(next) - Φ(current)

This naturally rewards moving closer, slowing down, and stabilizing simultaneously.

Let me also make the soft_landing_proxy a sum-based approach instead of product.

Wait, let me re-examine. The soft_landing_proxy nonzero_rate is 0.72% - that's still very sparse. The product of 4 continuous factors... let me check: dist_factor = max(0, 1-dist/1.0). At dist=0.5, dist_factor=0.5. speed_factor = max(0, 1-speed/0.6). At speed=0.3, speed_factor=0.5. angle_factor = max(0, 1-angle/0.4). At angle=0.2, angle_factor=0.5. contact_factor = min(1, (leg1+leg2)/2). If both legs are 0.5, contact_factor=0.5.

Product = 0.5*0.5*0.5*0.5 = 0.0625. That's small but non-zero. But the issue is that all 4 need to be simultaneously non-zero. If any one factor is zero, the whole product is zero.

The contact_factor is the likely bottleneck - the legs might not be in contact until the very last moment.

Let me try a different approach: make the landing proxy use a sum of weighted factors instead of product, so it provides signal even when not all conditions are met.

Actually, let me try the potential_based_shaping as the main signal, and simplify the landing proxy.

Let me design:

1. **potential_based_shaping** (main signal): Φ = -(dist + 0.8*speed + 0.5*abs(angle)), γ=0.99
   This replaces progress_delta_reward and part of stability_penalty

2. **stability_penalty** (reduced): Only for angular velocity, distance-gated

3. **soft_landing_proxy** (sum-based): Sum of individual factors instead of product, to increase nonzero rate

Let me implement this.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断与修改说明：
    # 问题：score=-115，100% early terminal。agent 在 progress_delta 引导下
    # 接近目标，但 crash 而非 soft landing。soft_landing_proxy 的 nonzero_rate=0.72%
    # 仍然太稀疏（乘积形式导致任一因子为零则整体为零）。
    #
    # 修改方案：
    # 1. 用 potential_based_shaping 替代 progress_delta_reward。
    #    势能函数 Φ = -(dist + 0.8*speed + 0.5*|angle|)，同时引导接近、减速、稳定。
    #    理论保证最优策略不变（Ng 1999），天然抗震荡。
    # 2. soft_landing_proxy 改为加权和形式（非乘积），提高 nonzero_rate。
    #    每个因子独立贡献梯度，不会因为某个条件不满足而完全消失。
    # 3. stability_penalty 保留角速度惩罚（距离门控），速度和角度已由 shaping 覆盖。
    # ============================================================
    
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    
    current_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    next_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    
    current_angle = abs(obs[4])
    next_angle = abs(next_obs[4])
    
    # ============================================================
    # 1. 主学习信号: potential_based_shaping
    #    势能 Φ = -(dist + 0.8*speed + 0.5*|angle|)
    #    shaping = γ * Φ(next) - Φ(obs), γ=0.99
    #    同时引导：接近目标 ↓、减速 ↓、姿态稳定 ↓
    # ============================================================
    gamma = 0.99
    phi_obs = -(current_dist + 0.8 * current_speed + 0.5 * current_angle)
    phi_next = -(next_dist + 0.8 * next_speed + 0.5 * next_angle)
    
    shaping_scale = 2.0
    potential_shaping = shaping_scale * (gamma * phi_next - phi_obs)

    # ============================================================
    # 2. 稳定约束: angular_vel_penalty（距离门控）
    #    速度和角度已由 shaping 覆盖，只保留角速度惩罚
    # ============================================================
    angular_vel = abs(next_obs[5])
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_dist / gate_radius)
    
    angular_vel_penalty_weight = 0.02
    angular_vel_penalty = -angular_vel_penalty_weight * angular_vel * distance_gate

    # ============================================================
    # 3. 任务完成proxy: soft_landing_proxy（加权和形式）
    #    用加权和替代乘积，每个因子独立贡献梯度
    #    提高 nonzero_rate，让 agent 逐步学会各个条件
    # ============================================================
    # 距离因子：dist < 0.8 时开始贡献
    dist_factor = max(0.0, 1.0 - next_dist / 0.8)
    # 速度因子：speed < 0.5 时开始贡献
    speed_factor = max(0.0, 1.0 - next_speed / 0.5)
    # 姿态角因子：angle < 0.3 时开始贡献
    angle_factor = max(0.0, 1.0 - next_angle / 0.3)
    # 接触因子
    contact_factor = min(1.0, (next_obs[6] + next_obs[7]) / 2.0)
    
    # 加权和（各因子权重不同，dist 最重要）
    landing_bonus_weight = 1.5
    soft_landing_proxy = landing_bonus_weight * (
        0.4 * dist_factor + 
        0.3 * speed_factor + 
        0.2 * angle_factor + 
        0.1 * contact_factor
    )

    # ============================================================
    # 4. 动作代价: energy_penalty（小权重）
    # ============================================================
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * engine_use

    # ============================================================
    # 总奖励
    # ============================================================
    total_reward = potential_shaping + angular_vel_penalty + soft_landing_proxy + energy_penalty

    components = {
        "potential_shaping": potential_shaping,
        "angular_vel_penalty": angular_vel_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

Wait, let me reconsider. The potential_based_shaping with phi = -(dist + 0.8*speed + 0.5*angle) means:

shaping = gamma * (-(next_dist + 0.8*next_speed + 0.5*next_angle)) - (-(current_dist + 0.8*current_speed + 0.5*current_angle))
= -(gamma*next_dist - current_dist) - 0.8*(gamma*next_speed - current_speed) - 0.5*(gamma*next_angle - current_angle)

With gamma close to 1, this is approximately:
≈ -(next_dist - current_dist) - 0.8*(next_speed - current_speed) - 0.5*(next_angle - current_angle)
= (current_dist - next_dist) + 0.8*(current_speed - next_speed) + 0.5*(current_angle - next_angle)

So it rewards: getting closer (+progress_delta), slowing down (+speed reduction), and stabilizing (+angle reduction). This is exactly what we want!

But I need to be careful about the scale. Let me set shaping_scale appropriately.

Actually, let me also think about whether this is too aggressive of a change. The previous version had progress_delta_reward with scale=1.0 giving mean=0.016. The potential_shaping will also include speed and angle terms, so it might be larger. Let me use shaping_scale=1.0 to start.

Let me also reconsider the soft_landing_proxy. With weighted sum, the max value would be landing_bonus_weight * (0.4+0.3+0.2+0.1) = landing_bonus_weight * 1.0 = 1.5. That's reasonable.

Let me finalize the code.

Actually, I realize I should also check - the previous version had progress_scale=1.0 and progress_delta_reward mean was 0.016. The potential_shaping with scale=1.0 would have similar magnitude for the distance component plus additional terms for speed and angle. Let me use shaping_scale=1.0 to keep things balanced.

Let me write the final version.

Hmm, actually one more thought. The soft_landing_proxy with weighted sum - the contact_factor weight is only 0.1, so even if contact is 0, the proxy can still be up to 1.5*0.9=1.35. This should give much higher nonzero_rate. Good.

Let me finalize.

Actually wait - I want to double check one thing. The previous soft_landing_proxy had nonzero_rate=0.72% even with the continuous product form. Let me think about why...

The product form: dist_factor * speed_factor * angle_factor * contact_factor

For this to be > 0, ALL four factors must be > 0 simultaneously. That means:
- dist < 1.0 (dist_factor > 0)
- speed < 0.6 (speed_factor > 0)
- angle < 0.4 (angle_factor > 0)
- contact_factor > 0 (at least one leg has some contact)

The contact_factor = min(1.0, (obs[6]+obs[7])/2.0). For this to be > 0, we need obs[6]+obs[7] > 0, meaning at least one leg has some contact.

The issue is likely that the agent is crashing before all 4 conditions are met simultaneously. The contact might only happen at the moment of crash, when speed and angle are bad.

With the weighted sum approach, even if contact=0, the agent still gets reward for being close, slow, and stable. This should provide much better gradient.

OK, I'm confident in this approach. Let me write the code.

Let me also think about whether I should keep the progress_delta_reward as a fallback... No, the potential_based_shaping with phi including distance already covers progress_delta. And it's theoretically cleaner.

One more thing - I want to make sure the shaping_scale is right. Let me use 1.5 to give it a bit more weight than the previous progress_delta_reward (which had scale=1.0 and mean=0.016).

Actually, let me just use 1.0 and see. The shaping includes speed and angle terms too, so it'll naturally be larger.

Let me finalize the code now.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断与修改说明：
    # 问题：score=-115，100% early terminal。agent 在 progress_delta 引导下
    # 接近目标，但 crash 而非 soft landing。soft_landing_proxy 的 nonzero_rate=0.72%
    # 仍然太稀疏（乘积形式导致任一因子为零则整体为零）。
    #
    # 修改方案：
    # 1. 用 potential_based_shaping 替代 progress_delta_reward。
    #    势能函数 Φ = -(dist + 0.8*speed + 0.5*|angle|)，同时引导接近、减速、稳定。
    #    理论保证最优策略不变（Ng 1999），天然抗震荡。
    # 2. soft_landing_proxy 改为加权和形式（非乘积），提高 nonzero_rate。
    #    每个因子独立贡献梯度，不会因为某个条件不满足而完全消失。
    # 3. stability_penalty 简化为仅角速度惩罚（距离门控），速度和角度已由 shaping 覆盖。
    # ============================================================
    
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    
    current_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    next_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    
    current_angle = abs(obs[4])
    next_angle = abs(next_obs[4])
    
    # ============================================================
    # 1. 主学习信号: potential_based_shaping
    #    势能 Φ = -(dist + 0.8*speed + 0.5*|angle|)
    #    shaping = γ * Φ(next) - Φ(obs), γ=0.99
    #    展开 ≈ (dist_curr-dist_next) + 0.8*(speed_curr-speed_next) + 0.5*(angle_curr-angle_next)
    #    同时引导：接近目标、减速、姿态稳定
    # ============================================================
    gamma = 0.99
    phi_obs = -(current_dist + 0.8 * current_speed + 0.5 * current_angle)
    phi_next = -(next_dist + 0.8 * next_speed + 0.5 * next_angle)
    
    shaping_scale = 1.5
    potential_shaping = shaping_scale * (gamma * phi_next - phi_obs)

    # ============================================================
    # 2. 稳定约束: angular_
