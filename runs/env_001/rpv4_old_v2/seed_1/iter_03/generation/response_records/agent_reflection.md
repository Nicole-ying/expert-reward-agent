# Response Record

1. `evidence`：iter2的 score=-104、len=301、terminated 19/20，contact_bonus 的 active_rate 仅 0.3%（死组件），fuel_penalty 的 magnitude_share 达 50.9% 且每步 -0.038，是最大负贡献源；progress 每步仅 +0.0073，无法驱动着陆。
2. `behavior_diagnosis`：agent 学会了长时间飞行（len 从 71 增至 301），但始终无法触发双腿稳定接触，导致 reward 由持续的燃料惩罚支配，最终以 crash/settle 结束且得分极低。
3. `signal_completeness`：缺少连续的近地下降引导和强着陆成功信号；progress 仅奖励向原点移动，不奖励着陆过程，contact_bonus 完全稀疏，无法形成有效的学习梯度。
4. `selected_level`：Level 2，因 contact_bonus 是死组件且其稀疏性已证明无法驱动着陆行为，需改变其数学形式。
5. `selected_intervention`：唯一目标组件 contact_bonus，将其从单纯的双腿接触稳定奖励改为“近地连续引导（y<1 时奖励下降速度、姿态、角速度）+接触时的高额稳定性奖励”。
6. `falsifiable_hypothesis`：引入连续近地奖励将为 agent 提供可学习的下降与姿态调整梯度，使其更可能触地并在接触后获得大额稳定奖励，从而改善得分并减少无效悬停。
7. `expected_next_round`：下一轮 contact_bonus 的 active_rate 应显著上升，per-step 量级增大，score 应趋近于 0 或转正，len 可能缩短，terminated 仍以多数但成功率应提升。
8. `main_risk`：连续近地奖励可能鼓励 agent 在低空低速悬停而不真正完成着陆，若接触奖励的梯度不够突出，可能出现“停在空中吃分”的行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 进展奖励（保持原样） ----------
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5
    progress_delta = dist_obs - dist_next

    speed_sq = vx_next**2 + vy_next**2
    gate_speed = 1.0 / (1.0 + 0.1 * speed_sq)
    gate_angle = 1.0 / (1.0 + 0.5 * abs(angle_next))
    gate_health = gate_speed * gate_angle
    progress_reward = 5.0 * progress_delta * gate_health

    # ---------- 重新设计的 contact_bonus：近地连续引导 + 接触高额奖励 ----------
    contact_bonus = 0.0
    descent_target_vy = -0.5   # 期望的下降速度
    vy_error = vy_next - descent_target_vy
    soft_factor = (2.718281828 ** (-(vy_error**2) / 0.1)) * \
                  (2.718281828 ** (-(vx_next**2) / 0.1)) * \
                  (2.718281828 ** (-(angle_next**2) / 0.05)) * \
                  (2.718281828 ** (-(angvel_next**2) / 0.05))

    if y_next < 1.0:
        contact_bonus = 5.0 * soft_factor   # 近地鼓励良好的下降与姿态

    if left_contact_next > 0.5 and right_contact_next > 0.5:
        contact_bonus += 30.0 * soft_factor # 接触时大额稳定性奖励

    # ---------- 燃料惩罚（保持原样） ----------
    fuel_cost = 0.05
    fuel_penalty = -fuel_cost if action != 0 else 0.0

    # ---------- 近地姿态惩罚（保持原样） ----------
    orientation_penalty = 0.0
    if y_next < 0.5:
        orientation_penalty = -0.5 * (angle_next**2) - 0.2 * (abs(angvel_next)**2)

    total_reward = progress_reward + contact_bonus + fuel_penalty + orientation_penalty

    components = {
        "progress": progress_reward,
        "contact_bonus": contact_bonus,
        "fuel_penalty": fuel_penalty,
        "orientation_penalty": orientation_penalty
    }
    return float(total_reward), components
```
