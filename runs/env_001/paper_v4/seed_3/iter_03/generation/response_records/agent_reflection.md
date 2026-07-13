# Response Record

`evidence`：上一轮 20/20 回合均在 1000 步截断、零次成功终止（settled），contact_bonus 占总 reward 幅值的 95.9%（episode_sum_mean=367.97），active_rate=76.2%，表明 agent 学会在目标附近保持双腿接触来持续收割该奖励，但从未真正减速至物理引擎休眠阈值；progress、velocity_penalty、orientation_penalty 三个组件合计仅占 4.1% 幅值，完全被淹没。

`behavior_diagnosis`：agent 已能到达目标附近并建立双腿接触，但它停在"接近目标+接触"的局部最优平台上，不断收割 contact_bonus 而不执行软着陆所需的最终减速与姿态稳定。所有回合跑满 1000 步，无一次触发 settled 终止。

`signal_completeness`：任务所需的过程引导（progress 趋向目标）、稳定约束（orientation_penalty）和速度约束（velocity_penalty）在形式上都有，但 contact_bonus 作为持续状态奖励（"在目标附近+接触"就给分），允许 agent 无需完成减速着陆即可获得几乎全部收益——缺少一个只有真正趋向 settled 才能最大化的信号。

`selected_level`：Level 2 — 证据符合 `state_to_improvement` 触发条件：agent 占据高回报状态即可持续获奖而不完成任务。将 contact_bonus 从"在好状态就给分"改为"状态越好（越接近 settled）给分越多"，在数学结构中嵌入速度和姿态质量因子。

`selected_intervention`：唯一修改 contact_bonus 组件，重命名为 `settling_quality`，在原有 contact × proximity 乘积上再乘一个 `settling_factor`：速度越低、倾角越小、角速度越小，该因子越接近 1；高速或大倾角时因子显著衰减。同步调整系数从 0.5 → 2.0 以匹配新值域。

`falsifiable_hypothesis`：新 reward 下，agent 无法仅靠"在目标附近+接触"获得满额奖励；必须在接触前提下降低速度与角速度才能最大化每步收益，这将推动策略越过当前局部最优，向 settled 状态逼近。预期截断率下降、出现成功终止。

`expected_next_round`：terminated 比例应从 0/20 上升；episode_length 均值应从 1000 下降；settling_quality 的 active_rate 保持较高但 episode_sum_mean 应下降（因为成功终止会缩短回合）；external score 应改善。

`main_risk`：settling_factor 中的速度衰减系数（5.0、2.0、1.0）可能过于苛刻，导致有效奖励过于稀疏，agent 退化为无法到达目标；若下一轮出现早期崩溃，需放宽 settling_factor 的分母常数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function with progress-based guidance and settling quality incentive.

    Components:
    - progress: rewards reducing distance to target center (0,0)
    - velocity_penalty: speed penalty gated by proximity to target
    - orientation_penalty: penalty for tilt and angular velocity
    - settling_quality: rewards dual-leg contact near target with low velocity and stable attitude
      (replaces persistent contact_bonus to prevent proxy farming without actual settling)
    """
    # Current state
    x_pos, y_pos = obs[0], obs[1]
    # Next state
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distances to target
    prev_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # 1. Progress reward: positive for moving toward target, negative for moving away
    progress_weight = 5.0
    progress_reward = progress_weight * (prev_dist - next_dist)

    # 2. Velocity penalty – active only when close to target
    proximity_factor = 1.0 / (1.0 + 10.0 * next_dist)
    vel_weight = 1.0
    velocity_penalty = -vel_weight * proximity_factor * (nx_vel ** 2 + ny_vel ** 2)

    # 3. Orientation stability – keep body upright and avoid spinning
    orient_weight = 0.5
    orientation_penalty = -orient_weight * (n_angle ** 2 + 0.2 * n_angvel ** 2)

    # 4. Settling quality – rewards contact + proximity + low velocity/attitude
    #    Replaces flat contact_bonus to prevent farming without actually settling.
    #    settling_factor approaches 1.0 only when velocity, angle, and angvel are all near zero.
    vel_magnitude = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    settling_factor = 1.0 / (1.0 + 5.0 * vel_magnitude + 2.0 * abs(n_angle) + 1.0 * abs(n_angvel))
    contact_product = left_contact * right_contact
    settling_weight = 2.0
    settling_quality = settling_weight * contact_product * proximity_factor * settling_factor

    total_reward = progress_reward + velocity_penalty + orientation_penalty + settling_quality

    components = {
        'progress': float(progress_reward),
        'velocity_penalty': float(velocity_penalty),
        'orientation_penalty': float(orientation_penalty),
        'settling_quality': float(settling_quality)
    }

    return float(total_reward), components
```
