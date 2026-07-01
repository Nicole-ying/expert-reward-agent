# Response Record

Let me analyze the situation carefully.

**Current state analysis:**

1. **score = -118.44** (very bad), **episode length = 71.9** (very short), **100% early termination** - the agent is dying/crashing very early.

2. **progress_delta_reward** has mean=0.162, ratio=1.0 - this is positive, meaning the agent IS moving toward the target. But it's dying before reaching it.

3. **stability_penalty** has mean=-0.006, ratio=-0.037 - very small, not dominating. The distance gate is working (it's very weak).

4. **soft_landing_bonus** has nonzero_rate=1.57% - barely ever triggers. The contact_factor (both feet on ground) is a hard binary gate that kills the bonus.

5. **energy_penalty** is negligible.

**The core problem:** The agent moves toward the target (positive progress) but falls over and dies early. The stability penalty is too weak to teach the agent to stay upright while moving. The distance gate makes it even weaker.

**What I changed from iter 1 to iter 2:** I reduced stability_penalty coefficients by ~20x (from -0.5, -0.3, -0.2 to -0.025, -0.015, -0.01) and added a distance gate. This made the penalty much weaker, and the agent got WORSE (score dropped from -110.68 to -118.44).

The problem is clear: **the stability penalty is too weak** to prevent the agent from falling over. The agent moves toward the target but tumbles and dies. I need to increase the stability penalty significantly, but in a smarter way - not a flat penalty that discourages all movement, but one that specifically penalizes the dangerous behaviors that cause falling.

Let me think about what to do:

1. The progress_delta_reward is working (positive mean) - keep it.
2. The stability_penalty is too weak - need to strengthen it.
3. The soft_landing_bonus never triggers because the agent dies before reaching the target - need to make it easier to trigger or focus on survival first.

Actually, looking more carefully: the agent dies at ~72 steps with score -118. The original env reward is -1.72 per step on average. So the agent is getting negative env reward (probably falling and getting -100 terminal penalty or something). The agent needs to learn to stay upright FIRST, then move toward target.

Let me try a different approach: **increase stability penalty significantly** but make it more nuanced. The previous version (iter 1) had stability_penalty mean=-0.242 which was too harsh. But now it's -0.006 which is too weak. Let me find a middle ground.

Also, the soft_landing_bonus with contact_factor as a hard binary gate is problematic. Let me make it continuous.

Let me also consider: the agent is dying early. The progress_delta_reward encourages moving toward target, but the agent falls over while doing so. I need stability to be strong enough to keep the agent upright.

Let me try:
- Increase stability penalty coefficients to something moderate
- Remove the distance gate (or make it much weaker) so stability applies everywhere
- Make soft_landing_bonus continuous (remove the binary contact_factor gate)

Actually, let me reconsider. The ratio_to_progress for stability_penalty in iter 1 was likely much higher (mean=-0.242). Let me check: progress_delta_reward mean was 0.160 in iter 1. So ratio was -0.242/0.160 = -1.51. That's too dominant.

Now it's -0.006/0.162 = -0.037. Too weak.

I need something in between. Let me try coefficients that give ratio around -0.3 to -0.5.

If progress mean ≈ 0.16, and I want stability mean ≈ -0.05 to -0.08 (ratio -0.3 to -0.5):
- angle_penalty: -0.1 * abs(body_angle) → mean abs(body_angle) maybe ~0.3? So -0.03
- angular_vel_penalty: -0.05 * abs(angular_vel) → mean maybe ~0.5? So -0.025
- speed_penalty: -0.03 * speed → mean speed maybe ~1.0? So -0.03
Total ≈ -0.085. That's ratio ≈ -0.53. Might work.

Let me also remove the distance gate since the agent dies early (far from target) and needs stability guidance everywhere.

For soft_landing_bonus: make contact_factor continuous too. Use `float(next_left_contact > 0.5) * float(next_right_contact > 0.5)` but maybe as a soft product.

Actually, let me make contact_factor also continuous: `(next_left_contact) * (next_right_contact)` since contact values are likely in [0,1] range.

Let me also increase the soft_landing_bonus weight since it barely triggers.

Let me write the new code:

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. Main learning signal: progress_delta_reward
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty - moderate strength, no distance gate
    # The agent dies early because it falls over while moving toward target
    # Need enough penalty to teach upright posture
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.1 * abs(next_body_angle)
    angular_vel_penalty = -0.05 * abs(next_angular_vel)
    speed_penalty = -0.03 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy: continuous product of bounded factors
    # Make contact_factor continuous too (use raw contact values)
    proximity_factor = 1.0 / (1.0 + 5.0 * next_dist)
    speed_factor = 1.0 / (1.0 + 5.0 * speed)
    angle_factor = 1.0 / (1.0 + 10.0 * abs(next_body_angle))
    angular_vel_factor = 1.0 / (1.0 + 5.0 * abs(next_angular_vel))
    # Continuous contact factor - use raw values as probabilities
    contact_factor = next_left_contact * next_right_contact  # continuous in [0,1]
    
    soft_landing_bonus = 10.0 * proximity_factor * speed_factor * angle_factor * angular_vel_factor * contact_factor
    
    # 4. Small energy penalty for using engines
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

Wait, but I need to be careful. Let me re-examine the problem.

The agent dies at step ~72 with score -118. The env reward is -1.72 per step. So total env reward over 72 steps ≈ -124. That's close to -118. So the env is giving very negative rewards (probably a big penalty for falling).

The progress_delta_reward is +0.16 per step, so over 72 steps that's about +11.5. The total reward is 0.165 per step, so over 72 steps ≈ +11.9. But the env reward is -1.72 per step = -124 total.

So the agent IS making progress (moving toward target) but the env is punishing it heavily (probably for falling/terminating badly). The custom reward is positive but the env reward is very negative.

This means the agent needs to learn to stay upright AND move toward target simultaneously. The stability penalty needs to be strong enough to teach upright behavior.

Let me also consider: maybe I should increase the progress_delta_reward scale to give more positive signal, or change the approach entirely.

Actually, let me look at this differently. The agent terminates early (100% of episodes < 150 steps). The env likely gives a large negative terminal reward for falling. The agent needs to survive longer.

To survive longer, the agent needs to:
1. Stay upright (stability)
2. Not crash into the ground

The progress_delta_reward is fine for guiding toward target. The stability_penalty needs to be strong enough to teach the agent to balance.

Let me go with moderate stability penalties and see.

Actually, I realize I should also consider: maybe the issue is that the agent is moving too fast and crashing. The speed_penalty could help with that. Let me keep my plan.

Let me also increase the soft_landing_bonus weight to 10.0 (from 5.0) since it's a product of 5 factors each in [0,1], so the product is very small.

Let me finalize my changes:

1. **Stability penalty**: Increase coefficients from (-0.025, -0.015, -0.01) to (-0.1, -0.05, -0.03) - roughly 4x stronger. Remove distance gate so it applies everywhere.
2. **Soft landing bonus**: Make contact_factor continuous (product of raw values instead of binary AND), increase weight from 5.0 to 10.0.
3. Keep everything else the same.

This is a focused change: adjusting stability penalty magnitude and making soft_landing_bonus more accessible.

## 诊断

**核心问题：Agent 在 72 步内 100% 提前终止，score -118。** 虽然 progress_delta_reward 为正（+0.16/步），说明 agent 在向目标移动，但它一边移动一边摔倒/坠毁。

**Iter 1 → Iter 2 的改动**：把 stability_penalty 系数从 (-0.5, -0.3, -0.2) 降到 (-0.025, -0.015, -0.01) 并加了距离门控。结果 score 从 -110.68 降到 -118.44，**变差了**。说明稳定性惩罚太弱，无法阻止 agent 摔倒。

**当前症状**：
- stability_penalty ratio=-0.037 → 太弱，几乎不起作用
- soft_landing_bonus nonzero_rate=1.57% → 几乎从不触发（agent 在到达目标前就死了）
- 100% 提前终止 → agent 需要更强的稳定性引导才能存活更久

**修改计划（一次改一个方面）**：
1. **稳定性惩罚系数提升约 4 倍**：从 (-0.025, -0.015, -0.01) → (-0.1, -0.05, -0.03)，并**移除距离门控**（agent 在远处也需要稳定性，否则还没到目标就摔了）
2. **软着陆奖励的 contact_factor 改为连续值**：用 `next_left_contact * next_right_contact` 代替二值 AND，让梯度更密集；同时权重从 5.0 提升到 10.0

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. Main learning signal: progress_delta_reward
    # Reward for moving closer to the target (origin in relative coordinates)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty - moderate strength, NO distance gate
    # Agent dies early (step ~72) because it falls over while moving.
    # Previous iteration had penalty too weak (ratio=-0.037).
    # Increasing coefficients ~4x to provide meaningful guidance.
    # Removing distance gate because agent needs stability everywhere,
    # not just near target - it dies before reaching target.
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.1 * abs(next_body_angle)
    angular_vel_penalty = -0.05 * abs(next_angular_vel)
    speed_penalty = -0.03 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy: continuous product of bounded factors
    # Changed contact_factor from binary AND to continuous product
    # of raw contact values, providing gradient even when only one foot touches.
    # Increased weight from 5.0 to 10.0 to compensate for product attenuation.
    proximity_factor = 1.0 / (1.0 + 5.0 * next_dist)  # bounded [0,1], high when near target
    speed_factor = 1.0 / (1.0 + 5.0 * speed)  # bounded [0,1], high when slow
    angle_factor = 1.0 / (1.0 + 10.0 * abs(next_body_angle))  # bounded [0,1], high when upright
    angular_vel_factor = 1.0 / (1.0 + 5.0 * abs(next_angular_vel))  # bounded [0,1], low angular vel
    
    # Continuous contact factor: product of raw values (assumed in [0,1])
    # This gives gradient even when only one foot contacts
    contact_factor = next_left_contact * next_right_contact  # continuous in [0,1]
    
    soft_landing_bonus = 10.0 * proximity_factor * speed_factor * angle_factor * angular_vel_factor * contact_factor
    
    # 4. Small energy penalty for using engines (action != 0)
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    # Combine rewards
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```
