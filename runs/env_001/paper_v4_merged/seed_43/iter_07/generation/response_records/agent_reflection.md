# Response Record

1. `evidence`：所有episode均early terminal（len≈68），score范围[-139.85, -89.94]极其负；angle_hinge_penalty active_rate=0%为僵尸组件；progress_shaping与shaped_progress占据89% magnitude但未阻止episode崩溃。
2. `behavior_diagnosis`：策略无法存活，频繁坠毁/出界；主信号progress_shaping在坠毁阶段因距离急剧增大而产生无界的大负奖励，导致episode早早被极端负值支配。
3. `signal_completeness`：缺失成功着陆事件奖励，且当前progress_shaping对退步采用无界线性惩罚，形同灾难性失败惩罚，信号配置不利于从失败中恢复。
4. `selected_level`：Level 2 — 结构变换，触发条件：无界→有界（progress_shaping的负分支在坠毁时爆炸，需bounding）。
5. `selected_intervention`：仅修改progress_shaping组件，从potential-based无界差分变为基于距离增量的bounded improvement（进步系数0.5，退步系数0.05），以压制退步时的灾难性惩罚。
6. `falsifiable_hypothesis`：将退步惩罚降低至0.05倍后，坠毁时的负贡献幅度从爆炸变为温和，总reward负幅应大幅缩小，episode存活步数可能回升。
7. `expected_next_round`：score应显著上升（负值减小），len均值回弹，progress_shaping的episode_sum_mean可能略微下降但其负贡献份额消失，angle_hinge_penalty仍为0。
8. `main_risk`：退步惩罚过弱可能使agent在远离目标时无法获得足够负反馈，导致策略徘徊而不是坚定靠近目标。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current observation
    x = float(obs[0])
    y = float(obs[1])

    # Next observation
    nx = float(next_obs[0])
    ny = float(next_obs[1])
    nvx = float(next_obs[2])
    nvy = float(next_obs[3])
    nangle = float(next_obs[4])
    left_contact = float(next_obs[6])
    right_contact = float(next_obs[7])

    # ---------- 1. Bounded distance-based progress ----------
    dist_obs = (x * x + y * y) ** 0.5
    dist_next = (nx * nx + ny * ny) ** 0.5
    dist_delta = dist_obs - dist_next   # positive = getting closer

    progress_pos = 0.5 * max(0.0, dist_delta)
    progress_neg = 0.05 * min(0.0, dist_delta)
    progress_shaping = progress_pos + progress_neg

    # ---------- 2. Landing speed gate ----------
    proximity_factor = max(0.0, 1.0 - dist_next / 0.5)
    speed_next = (nvx * nvx + nvy * nvy) ** 0.5
    speed_cost_input = speed_next * proximity_factor
    landing_speed_gate = 1.0 / (1.0 + 5.0 * speed_cost_input)

    shaped_progress = progress_shaping * landing_speed_gate

    # ---------- 3. Action cost ----------
    action_cost = -0.01 * (0.0 if action == 0 else 1.0)

    # ---------- 4. Landing contact bonus ----------
    contact_sum = left_contact + right_contact
    contact_factor = contact_sum / 2.0
    proximity = max(0.0, 1.0 - dist_next / 0.8)
    landing_contact_reward = 0.2 * contact_factor * proximity

    # ---------- 5. Angle hinge penalty (kept as is) ----------
    angle_abs = abs(nangle)
    angle_excess = max(0.0, angle_abs - 0.3)
    angle_hinge_penalty = -0.03 * angle_excess

    total_reward = shaped_progress + action_cost + landing_contact_reward + angle_hinge_penalty

    components = {
        "progress_shaping": progress_shaping,
        "shaped_progress": shaped_progress,
        "action_cost": action_cost,
        "landing_contact_reward": landing_contact_reward,
        "angle_hinge_penalty": angle_hinge_penalty
    }

    return float(total_reward), components
```
