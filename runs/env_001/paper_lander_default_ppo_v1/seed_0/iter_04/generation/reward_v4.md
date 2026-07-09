## Analysis

1. `evidence`：v3 将 sparse soft_landing_proxy 换成连续 approach_quality_reward 后 score 从 -94 跳升到 110，但现在该组件 episode_sum_mean=493 占 99.7%，progress_reward 仅 1.35，terminated=9/20 说明近半 episode 未能着陆即被截断，score_range 高达 290 但有负分 -29，存在"能靠近但不愿着陆"的证据。

2. `behavior_diagnosis`：agent 学会了在目标附近保持低速、直立姿态来持续收割 per-step 状态奖励（~0.8/step × 613 步 ≈ 493），而非完成着陆；因为着陆会导致 episode 终止、奖励流中断，strategic hovering 是当前奖励结构下的理性策略。

3. `signal_completeness`：缺少"着陆完成"的有效激励——progress_reward 提供靠近信号、stability_penalty 提供平滑信号，但没有任何组件奖励从靠近状态**过渡到着陆完成**这一步，approach_quality_reward 作为纯状态值奖励反而惩罚提前着陆。

4. `selected_level`：Level 2，触发条件为"占据好状态即可持续获奖"（state_to_improvement）和"proxy 提高但外部任务未完成"（dense_to_task_event 的模式重叠）。

5. `selected_intervention`：将 `approach_quality_reward` 从**绝对状态值**变换为**势能差塑形**（potential-based shaping）：reward = w × (Φ(s') − Φ(s))，其中 Φ 保持相同的 quality 因子乘积。w 设为 80 匹配新值域。

6. `falsifiable_hypothesis`：势能差塑形使总累积奖励与路径长度解耦——hovering 时 Φ 不变 → 奖励为 0，只有真正改善状态（靠近、减速、扶正、最终着陆）时才获得正奖励；着陆后 episode 自然终止，agent 不再有激励延迟着陆。

7. `expected_next_round`：episode length 应下降（不再徘徊），terminated 比例上升（更多着陆），approach_quality_reward 的 episode_sum_mean 应从 493 大幅降至约 50–70（受 Φ 上限约束），score 应上升。

8. `main_risk`：势能差在 Φ 接近饱和时梯度消失，可能导致着陆前最后阶段的精细控制信号不足；若发生则需在下一轮增加终端附近的梯度增强（如更陡的 proximity 函数或接触门控 bonus）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: state_to_improvement — replace absolute approach_quality_reward
    with potential-based shaping: reward = w * (Φ(s') - Φ(s)).
    Φ = proximity * speed_factor * angle_factor, same quality factors as v3.
    Coefficient w=80 matches the new value range (total bounded by ~80).
    """
    # -- Distance to target
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # -- 1. Progress delta (unchanged)
    delta_dist = dist_current - dist_next
    w_progress = 1.0
    progress_reward = w_progress * delta_dist

    # -- 2. Stability penalty (unchanged)
    w_vel    = 0.001
    w_angle  = 0.001
    w_angvel = 0.0001

    stability_penalty = (
        -w_vel    * (abs(next_obs[2]) + abs(next_obs[3]))
        -w_angle  * abs(next_obs[4])
        -w_angvel * abs(next_obs[5])
    )

    # -- 3. Landing potential shaping (replaces approach_quality_reward)
    # Quality factors for current state
    proximity_curr    = 1.0 / (1.0 + 5.0 * dist_current)
    speed_curr        = abs(obs[2]) + abs(obs[3])
    speed_factor_curr = 2.0 / (2.0 + speed_curr)
    angle_factor_curr = 1.0 / (1.0 + abs(obs[4]))
    potential_current = proximity_curr * speed_factor_curr * angle_factor_curr

    # Quality factors for next state
    proximity_next    = 1.0 / (1.0 + 5.0 * dist_next)
    speed_next        = abs(next_obs[2]) + abs(next_obs[3])
    speed_factor_next = 2.0 / (2.0 + speed_next)
    angle_factor_next = 1.0 / (1.0 + abs(next_obs[4]))
    potential_next = proximity_next * speed_factor_next * angle_factor_next

    w_potential = 80.0
    landing_shaping_reward = w_potential * (potential_next - potential_current)

    # -- Total reward
    total_reward = progress_reward + stability_penalty + landing_shaping_reward

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward
    }

    return float(total_reward), components
```