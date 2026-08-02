1. `evidence`：当前最终得分-87.19，episode长度143.7，全部terminated，其中早期终止9/20且得分<-50；progress_shaping和shaped_progress活跃率100%但未能阻止坠毁；landing_contact_reward活跃率11.9%贡献26.5%但稀疏；action_cost活跃45.9%负贡献13.8%；body_angle和angular_velocity观测维度完全未使用；累积记录中当前骨架从iter4至iter5未刷新best。

2. `behavior_diagnosis`：飞行器学会了朝目标垫中心靠近，但在接近阶段姿态失控，可能发生过大倾斜导致主体触地坠毁，或高速撞击地面，导致大量早期失败和很高的负分。

3. `signal_completeness`：已有进度引导、速度门控和接触奖励，但缺少对机身倾角的约束信号，无法阻止危险的大角度姿态，进而导致坠毁。

4. `selected_level`：Level 2 — 信号覆盖存在缺失（角度约束），需要添加一个新组件，属于结构变换。

5. `selected_intervention`：新增`angle_hinge_penalty`组件，对机身角度的绝对值超过0.3 rad的部分施加线性惩罚，系数0.03，引导飞行器保持水平姿态，避免触地坠毁。

6. `falsifiable_hypothesis`：增加角度惩罚后，策略会减少大幅倾斜行为，从而降低坠毁率，早期终止次数和负分应减少，整体得分提升。

7. `expected_next_round`：新组件`angle_hinge_penalty`活跃率应在俯仰较大时非零；平均每步惩罚约在-0.003左右；整体得分应上升（负得分的绝对值变小），episode_length可能保持或小幅增加，early_terminal占比下降。

8. `main_risk`：角度惩罚可能使飞行器过度保守，不敢进行必要的姿态调整，导致无法有效减速或对齐，反而在更长徘徊后坠毁，需监控episode_length和得分变化。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current observation
    x = float(obs[0])
    y = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])

    # Next observation
    nx = float(next_obs[0])
    ny = float(next_obs[1])
    nvx = float(next_obs[2])
    nvy = float(next_obs[3])
    nangle = float(next_obs[4])
    left_contact = float(next_obs[6])
    right_contact = float(next_obs[7])

    # ---------- 1. Main learning signal: potential-based shaping ----------
    dist_obs = (x * x + y * y) ** 0.5
    dist_next = (nx * nx + ny * ny) ** 0.5
    speed_obs = (vx * vx + vy * vy) ** 0.5
    speed_next = (nvx * nvx + nvy * nvy) ** 0.5

    alpha = 0.5
    potential_obs = -(dist_obs + alpha * speed_obs)
    potential_next = -(dist_next + alpha * speed_next)
    progress_shaping = potential_next - potential_obs

    # ---------- 2. Landing speed gate ----------
    proximity_factor = max(0.0, 1.0 - dist_next / 0.5)
    speed_cost_input = speed_next * proximity_factor
    landing_speed_gate = 1.0 / (1.0 + 5.0 * speed_cost_input)

    shaped_progress = progress_shaping * landing_speed_gate

    # ---------- 3. Efficiency: action penalty ----------
    action_cost = -0.01 * (0.0 if action == 0 else 1.0)

    # ---------- 4. Landing contact bonus ----------
    contact_sum = left_contact + right_contact
    contact_factor = contact_sum / 2.0
    proximity = max(0.0, 1.0 - dist_next / 0.8)
    landing_contact_reward = 0.2 * contact_factor * proximity

    # ---------- 5. Angle hinge penalty (new) ----------
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