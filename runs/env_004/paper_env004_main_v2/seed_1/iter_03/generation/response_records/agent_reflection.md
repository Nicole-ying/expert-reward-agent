# Response Record

`evidence`：V2 在 V1 基础上新增 height_reward（w=0.5, target=1.0, tolerance=0.5），该组件 episode_sum_mean=238.29（magnitude_share=51.4%, active_rate=100%），forward_progress 从 V1 的 dominant 信号骤降至 episode_sum_mean=206.18（active_rate=58.9%），episode_length 从 1000 降至 791.5，terminated 从 0 升至 6/20。height_reward 的恒定目标高度破坏了跳跃步态的自然高度振荡，策略为维持 1.0 高度而牺牲了前进速度。

`behavior_diagnosis`：V2 策略被 height_reward 主导，学会了维持固定站立高度而非动态跳跃前进，导致前进速度崩溃、部分 episode 提前跌倒终止。V1 策略稳定存活满 1000 步、以中等速度前进，但距离 target 仍有 ~20% 缺口，可能已收敛到"安全但不够快"的局部最优步态。

`signal_completeness`：V1 已具备前进引导（forward_progress）和稳定性约束（torso_angle + angular_velocity penalties），职责基本完备且可达。缺失的是推动策略突破"安全速度平台"、追求更高前进速度的边际激励。

`selected_level`：Level 1 — V1 骨架已验证有效（83% target），仅需一个微小组件推动速度最大化。height_reward 已在上轮证明有害，本轮不重复高度相关修改。

`selected_intervention`：在 V1 基础上新增 `forward_speed_bonus`：对小速度几乎无影响、对大速度提供边际递增奖励的二次项 `w_fwd2 * max(0, next_obs[5])**2`，w_fwd2=0.1。其他组件与 V1 完全一致。

`falsifiable_hypothesis`：二次速度奖励为高速步态提供额外梯度，应推动策略从 V1 的"安全中等速度"向"更快跳跃"方向探索，在保持稳定存活的前提下提升 episode_sum_mean 的 forward_progress 总量，从而缩小与 target 的差距。

`expected_next_round`：score 应高于 V1 的 3168，episode_length 保持接近 1000，terminated 保持接近 0，forward_progress 的 episode_sum_mean 应显著上升，forward_speed_bonus 在高速度段应有非零贡献但 magnitude_share 不应过大。

`main_risk`：二次项在高速度时增长较快，若 w_fwd2 过大可能促使策略采取过于激进的步态导致不稳定和更多跌倒；当前 w_fwd2=0.1 对应速度 3 时 bonus=0.9（约线性项的 30%），属于温和增量，但若下一轮出现 terminated 上升则需降系数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V3 reward: V1 base + quadratic forward speed bonus.
    Restores the successful V1 structure and adds a mild bonus for high forward velocity
    to push the agent beyond a "safe but moderate" speed plateau.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress
    w_fwd2 = 0.1         # quadratic bonus for higher speeds
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty

    # Main learning signal: forward velocity (clipped to non-negative)
    v_fwd = max(0.0, next_obs[5])
    forward_progress = w_fwd * v_fwd

    # Quadratic bonus: disproportionately rewards higher forward speeds
    forward_speed_bonus = w_fwd2 * (v_fwd ** 2)

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    total_reward = forward_progress + forward_speed_bonus + torso_angle_penalty + torso_angular_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "forward_speed_bonus": forward_speed_bonus,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty,
    }

    return float(total_reward), components
```
