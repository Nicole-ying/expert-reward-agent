1. `evidence`：19/20 episode被截断，平均975步，`goal_proximity`以-258的episode_sum_mean占magnitude_share 92.2%，这是持续状态惩罚在长episode中的无界累积；`contact_reward` active_rate仅0.5%，几乎不触发；速度/姿态惩罚尺度很小，无结构性异常。

2. `behavior_diagnosis`：agent长时间存活但始终远离目标垫（平均dist_sq≈258），在远处徘徊或悬停，无法接近目标，导致`-dist_sq`每步持续累积巨额负奖励，最终得分-51。

3. `signal_completeness`：当前缺少任何进展信号——`goal_proximity`是纯状态值，只惩罚"处于远处"而不奖励"靠近目标"；contact reward过于稀疏(0.5%)无法引导着陆；没有settlement bonus；职责层面缺少progress-based引导。

4. `selected_level`：Level 2 `state_to_improvement`——`goal_proximity = -dist_sq`是持续状态惩罚，在长episode中无界累积并主导整个奖励，证据直接否定当前数学形态；需将状态值改为状态改善量（势能差）。

5. `selected_intervention`：唯一修改`goal_proximity`组件，从`-w_goal * dist_sq`（持续状态惩罚）变为`w_goal * (dist_sq - next_dist_sq)`（势能差/进度奖励），w_goal保持1.0；其他组件（velocity_penalty、orientation_penalty、contact_reward及其proximity门控结构）完全不变。

6. `falsifiable_hypothesis`：改为势能差后，goal_proximity的episode_sum_mean将不再随episode长度线性累积，而是有界（总和=初始dist_sq-最终dist_sq）；agent每步获得靠近即正、远离即负的即时梯度，应能学会向目标移动，逐步降低最终距离，从而让contact_reward有机会触发。

7. `expected_next_round`：`goal_proximity`的`episode_sum_mean`应从-258大幅上升（向零或正值靠近），`magnitude_share`显著下降；episode_length可能缩短（agent学会接近目标后触发终止）；若接近成功，`contact_reward`的`active_rate`应从0.5%上升；`score`应从负值向零或正值改善。

8. `main_risk`：纯势能差在agent徘徊时净奖励接近零，可能使总奖励信号过弱，其他组件（速度/姿态惩罚虽小）在无进展时成为唯一驱动力，可能导致agent采取保守策略（如完全不动以规避速度惩罚）而非探索接近目标。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    x = obs[0]            # horizontal position relative to target pad
    y = obs[1]            # vertical position relative to target pad
    vx = obs[2]           # horizontal velocity
    vy = obs[3]           # vertical velocity
    angle = obs[4]        # body angle
    ang_vel = obs[5]      # angular velocity
    left_contact = obs[6] # left support leg contact (0 or 1)
    right_contact = obs[7]# right support leg contact (0 or 1)

    next_x = next_obs[0]
    next_y = next_obs[1]

    # Hyperparameters
    w_goal = 1.0
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 3.0

    # Distance to target center (squared) for current and next state
    dist_sq = x**2 + y**2
    next_dist_sq = next_x**2 + next_y**2

    # Narrow proximity gate for contact reward: only reward leg contact very close to target
    proximity_narrow = 1.0 / (1.0 + 5.0 * dist_sq)

    # Wide proximity gate for velocity penalty: provide deceleration feedback earlier on approach
    proximity_wide = 1.0 / (1.0 + 1.5 * dist_sq)

    # 1. Main progress: potential-based improvement reward (positive when approaching target)
    goal_proximity = w_goal * (dist_sq - next_dist_sq)

    # 2. Soft landing velocity penalty: active over a wider approach zone
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity_wide

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: reward both legs touching the pad, gated by narrow proximity
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    contact_reward = w_contact * both_legs_contact * proximity_narrow

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components
```