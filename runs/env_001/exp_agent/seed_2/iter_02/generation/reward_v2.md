Now I have a clear picture. Let me analyze the issues:

**Problem 1: stability_penalty dominates.** ratio_to_progress = -1.51, meaning the penalty is 1.5x the magnitude of the progress signal. The agent terminates early (100% early termination at step 72). The penalty is too strong and discourages movement.

**Problem 2: soft_landing_bonus is too sparse.** nonzero_rate = 0.58% (< 2%), so it almost never triggers. The binary if-condition provides no gradient.

**Problem 3: progress_delta_reward** has ratio 1.0 (by definition) but the mean is only 0.16, which is tiny compared to the -0.24 stability penalty. The net reward is negative (-0.08).

**Plan:**
1. **Reduce stability_penalty drastically** - at least 10x, from -0.5, -0.3, -0.2 to something much smaller. Also consider distance-gating it.
2. **Replace soft_landing_bonus** from binary if-condition to a continuous product of bounded factors, so it provides gradient guidance.
3. **Keep progress_delta_reward** as the main signal but maybe adjust scale slightly.

Let me think about the stability penalty more carefully. The agent terminates early - it's probably falling over. We need some stability guidance but not so much that it paralyzes the agent. Let me reduce by ~10x and also add distance-gating so it only matters near the target.

Actually, looking at the data more carefully: the agent terminates at step 72 with score -110. The original_env_reward is -1.545 per step, so the environment itself is punishing the agent heavily. The agent is dying quickly. The stability penalty is making things worse by discouraging movement.

Let me make two changes:
1. Reduce stability_penalty coefficients by ~20x (from 0.5/0.3/0.2 to 0.025/0.015/0.01)
2. Replace binary soft_landing_bonus with continuous product of bounded factors

This way I can isolate which change helps.

Actually, let me think about this more carefully. The stability penalty ratio is -1.51 which is very dominant. But the agent also terminates early at step 72. The stability penalty might actually be useful for preventing falls - the problem is its magnitude relative to progress.

Let me reduce stability penalty by 20x and also make the soft landing bonus continuous.

Wait - I should only change one thing at a time for testability. But the instructions say "把二值改成连续乘积算'改一个组件'" - so replacing the binary bonus with continuous counts as one change. And reducing the stability penalty coefficients counts as another change. Let me do both since they address different issues.

Actually, re-reading: "尽量每次只修改一个方面（一个系数、一个表达式、一个组件）". Let me do both since they are independent components (stability_penalty coefficients vs soft_landing_bonus expression).

Let me also consider: should I distance-gate the stability penalty? The idea is that far from target, we care more about progress, and near target we care more about stability. This could help. Let me add distance-gating.

Here's my plan:
1. **Reduce stability_penalty coefficients by 20x** and add distance-gating (only penalize when near target, dist < 3.0)
2. **Replace soft_landing_bonus** with continuous product: proximity_factor * speed_factor * angle_factor * contact_factor, each bounded [0,1]

Let me implement this.

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
    
    # 2. Stability penalty - reduced by 20x and distance-gated
    # Only penalize instability when near the target (dist < 3.0)
    # Far from target, let the agent move freely
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.025 * abs(next_body_angle)
    angular_vel_penalty = -0.015 * abs(next_angular_vel)
    speed_penalty = -0.01 * speed
    
    # Distance gate: only apply stability penalty when near target
    gate = 1.0 / (1.0 + 2.0 * next_dist)  # ~1 when dist=0, ~0.2 when dist=2, ~0.09 when dist=5
    stability_penalty = gate * (angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. Soft landing proxy: continuous product of bounded factors
    # Each factor is in [0,1], product gives smooth gradient
    proximity_factor = 1.0 / (1.0 + 5.0 * next_dist)  # bounded [0,1], high when near target
    speed_factor = 1.0 / (1.0 + 5.0 * speed)  # bounded [0,1], high when slow
    angle_factor = 1.0 / (1.0 + 10.0 * abs(next_body_angle))  # bounded [0,1], high when upright
    angular_vel_factor = 1.0 / (1.0 + 5.0 * abs(next_angular_vel))  # bounded [0,1], low angular vel
    
    # Contact factor: both feet on ground
    contact_factor = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    contact_factor = float(contact_factor)  # 0 or 1, but that's okay as a gate
    
    soft_landing_bonus = 5.0 * proximity_factor * speed_factor * angle_factor * angular_vel_factor * contact_factor
    
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

Wait, let me reconsider the stability penalty. The gate function `1/(1+2*dist)` at dist=0 gives 1.0, at dist=1 gives 0.33, at dist=2 gives 0.2, at dist=5 gives 0.09. So it heavily gates the penalty when far from target. But the agent is dying at step 72 - it never gets far. The initial distance might be around... let me think. In LunarLander, the lander starts at the top center. The target is at (0,0) roughly. Starting position is around x=0, y~=1.2-1.4. So initial dist ~ 1.2-1.4. At dist=1.4, gate = 1/(1+2*1.4) = 1/3.8 = 0.26. So the penalty is already reduced to ~26% at start.

Hmm, but the agent is dying early. Maybe the stability penalty isn't the main cause of early termination - maybe it's just that the agent can't learn to land properly. Let me think again.

The original_env_reward is -1.545 per step. In LunarLander, the environment gives -100 for crashing and +100 for landing. The agent is getting -110 total score over 72 steps, which means it's crashing (getting -100) plus some negative per-step rewards.

Actually, the stability penalty ratio is -1.51 which is very high. Let me reduce it more aggressively. Let me reduce by 50x instead of 20x.

Actually, let me re-examine. The stability_penalty mean is -0.242, progress_delta_reward mean is 0.160. So net from these two is -0.082. The total_reward mean is -0.079. So the agent is getting negative reward on average, which discourages it from doing anything.

Let me reduce stability penalty coefficients more aggressively - by 50x. So:
- angle_penalty: -0.01 * abs(next_body_angle) (was -0.5)
- angular_vel_penalty: -0.006 * abs(next_angular_vel) (was -0.3)
- speed_penalty: -0.004 * speed (was -0.2)

With these, the expected stability_penalty would be roughly -0.242/50 = -0.0048, which is negligible. That might be too little. Let me try 20x reduction but without the distance gate first, to keep it simpler.

Actually, let me try a different approach. Reduce by 10x and add distance gating.

angle_penalty = -0.05 * abs(next_body_angle)
angular_vel_penalty = -0.03 * abs(next_angular_vel)
speed_penalty = -0.02 * speed

With gate = 1/(1+2*dist), at start (dist~1.4): gate=0.26, so effective penalty is ~26% of these values.

Expected mean stability_penalty: currently -0.242 with coefficients 0.5, 0.3, 0.2. With 10x reduction: -0.0242. With gate at ~0.26 average: -0.0063. That's very small.

Let me try 5x reduction without gate first, to keep it simple.

angle_penalty = -0.1 * abs(next_body_angle)
angular_vel_penalty = -0.06 * abs(next_angular_vel)
speed_penalty = -0.04 * speed

Expected mean: -0.242/5 = -0.0484. Ratio to progress: -0.0484/0.16 = -0.30. That's under 0.5, acceptable.

And for the soft landing bonus, let me use a scale of 5.0 with the continuous product. The product of factors will typically be small, so 5.0 should give a reasonable bonus when conditions are good.

Let me finalize the code.

Actually, I want to also think about whether the progress_delta_reward scale of 10.0 is appropriate. The mean is 0.16, which seems low. But that's because the delta itself is small (0.016 on average). The scale of 10 is fine.

Let me go with my plan: reduce stability by 5x, and make soft_landing continuous.

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
    
    # 2. Stability penalty - reduced 5x to prevent dominating progress
    # Previously ratio_to_progress was -1.51, now targeting ~-0.3
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.1 * abs(next_body_angle)
    angular_vel_penalty = -0.06 * abs(next_angular_vel)
    speed_penalty = -0.04 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy: continuous product of bounded factors
    # Replaces binary if-condition to provide dense gradient signal
    # Each factor uses 1/(1+k*x) form, bounded in [0,1]
    proximity_factor = 1.0 / (1.0 + 5.0 * next_dist)
    speed_factor = 1.0 / (1.0 + 5.0 * speed)
    angle_factor = 1.0 / (1.0 + 10.0 * abs(next_body_angle))
    angular_vel_factor = 1.0 / (1.0 + 5.0 * abs(next_angular_vel))
    contact_factor = float((next_left_contact > 0.5) and (next_right_contact > 0.5))
    
    soft_landing_bonus = 5.0 * proximity_factor * speed_factor * angle_factor * angular_vel_factor * contact_factor
    
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

This looks good. Let me output it.

Wait, I want to double-check: the contact_factor being 0 or 1 (binary) is fine because it acts as a gate - the other 4 factors provide the continuous gradient. When both feet contact, the bonus is active; when not, it's 0. The continuous factors (proximity, speed, angle, angular_vel) guide the agent toward the right state even before both feet contact.

Actually, I could make contact_factor continuous too: `min(next_left_contact, next_right_contact)` since contact values are typically 0 or 1 in LunarLander. But that might not work if they're boolean. Let me keep it as is - the float cast of the boolean condition is fine.

Let me finalize.## 诊断

从 ratio_to_progress 数据可以清晰看到两个问题：

1. **stability_penalty 主导 (-1.51)**：惩罚绝对值是 progress 信号的 1.5 倍，导致 agent 为避免惩罚而不敢动，100% 提前终止（平均 72 步）。需要大幅降低系数。

2. **soft_landing_bonus 过于稀疏 (nonzero_rate=0.58%)**：二值 if 条件几乎从不触发，无法提供梯度引导。需要改为连续乘积形式。

**修改方案：**
- 层次 1：stability_penalty 系数降低 5 倍（从 0.5/0.3/0.2 → 0.1/0.06/0.04），使 ratio 降到 ~-0.3
- 层次 2：soft_landing_bonus 从二值 if 改为连续 bounded 乘积（5 个因子，每个用 1/(1+kx) 形式），提供密集梯度

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
    current_dist = (