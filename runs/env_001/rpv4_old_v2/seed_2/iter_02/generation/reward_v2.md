1. `evidence`：当前奖励函数在全部20个 episodes 上得分为 -120.998945，所有回合均被提前终止（无时间截断），平均长度仅68.35步；组件 active_rate 显示 success_proxy（0.1%）和 contact_quality（0.2%）几乎为死信号，而 goal_proximity、velocity_penalty、angle_penalty 三大惩罚项占据了99%以上的 signed_share，且在每步步长下均为负数，表明代理被持续的负向惩罚主导。
2. `behavior_diagnosis`：代理很可能因强烈的距离、速度和姿态惩罚而采取保守但无效的动作（如犹豫或过早坠毁），未能学会向目标垫稳定降落；缺乏任何可逐步触发的正向引导信号，使其无法在早期获得进展反馈，导致所有尝试均以失败终止。
3. `signal_completeness`：任务所需的渐进引导（靠近目标、减速、调平）当前只有负向惩罚，缺少连续、可逐步激活的正向接近奖励或完成度信号；必要的安全约束（速度、姿态）虽有惩罚但已全覆盖；现有的 success_proxy 因激活条件过于严苛（仅在高度完美时触发乘积）而成为死组件，无法提供可学习的完成关系。
4. `selected_level`：Level 2 – 修改目标组件的数学形式。因为 success_proxy 的 active_rate 仅 0.1%，单纯调整权重（Level 1）无法改变激活频率；必须将其从稀疏的乘积门控转换为更容易触发、提供连续梯度的加性奖励，同时保持权重合理以避免淹没原有惩罚。
5. `selected_intervention`：仅修改 `success_proxy` 组件。将其公式从基于几何平均的乘积门控改为对位置、速度、姿态、接触状态的线性衰减得分加和，每一项独立给出0到1之间的正向奖励，之和乘以一个调低的权重（w_proxy 由 20.0 降至 2.0），使得在非完美状态下也能提供逐步增强的正向拉力。
6. `falsifiable_hypothesis`：引入加性连续奖励后，代理会在训练中更早收到“接近成功”的弱正向信号，从而减少对纯惩罚的回避，逐步学会向目标垫移动并降低速度；预期成功代理的 active_rate 显著上升，episode 长度延长，score 提高（负值减小），且 terminated 比例可能下降或出现部分时间截断。
7. `expected_next_round`：success_proxy 的 active_rate 应至少提升至 10% 以上，episode 长度可能增长（如超过 100 步），score 均值应向正向移动（例如 -100 以上），且可能首次出现 truncated episodes。
8. `main_risk`：加性得分可能使代理在远距离时仍然获得少量正值，从而降低了对绝对位置偏差的惩罚力度，导致代理滞留在中距离区域进行微小调整而非直接完成最终着陆，延长了到达时间或产生异常悬停行为；此外，若 w_proxy 仍过大，可能使 total reward 过早转为正值，破坏先期惩罚的引导作用。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, which represents the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad (height)
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact flag (0 or 1)
    right_c = next_obs[7]   # right leg contact flag (0 or 1)

    # ------------------------------------------------------------------
    # Component A – goal proximity + soft landing  (main learning signal)
    # ------------------------------------------------------------------
    w_prox  = 2.0
    w_vel   = 1.0
    w_ang   = 2.0

    comp_prox  = w_prox * ( -abs(x) - abs(y) )
    comp_vel   = w_vel  * ( -vx**2 - vy**2 )
    comp_angle = w_ang  * ( -angle**2 - angvel**2 )

    # ------------------------------------------------------------------
    # Component B – contact quality (stability constraint)
    # ------------------------------------------------------------------
    w_contact = 10.0
    pos_thresh = 0.3
    vel_thresh = 0.3
    gate_x   = max(0.0, 1.0 - abs(x) / (pos_thresh + 1e-8))
    gate_y   = max(0.0, 1.0 - abs(y) / (pos_thresh + 1e-8))
    gate_vx  = max(0.0, 1.0 - abs(vx) / (vel_thresh + 1e-8))
    gate_vy  = max(0.0, 1.0 - abs(vy) / (vel_thresh + 1e-8))
    leg_contact = left_c * right_c
    comp_contact = w_contact * leg_contact * gate_x * gate_y * gate_vx * gate_vy

    # ------------------------------------------------------------------
    # Component C – success proxy (task‑completion continuous signal)
    # ------------------------------------------------------------------
    # Changed from geometric mean of near‑perfect conditions to an additive,
    # continuous reward that activates gradually as each condition improves.
    # This provides a smoother gradient for the agent to follow.
    w_proxy = 2.0   # reduced from 20.0 to avoid overwhelming penalties

    # Distance thresholds that produce a score from 1 (perfect) down to 0
    score_x     = max(0.0, 1.0 - abs(x) / 1.0)
    score_y     = max(0.0, 1.0 - abs(y) / 1.0)
    score_vx    = max(0.0, 1.0 - abs(vx) / 0.5)
    score_vy    = max(0.0, 1.0 - abs(vy) / 0.5)
    score_angle = max(0.0, 1.0 - abs(angle) / 0.3)
    score_contact = (left_c + right_c) * 0.5   # [0,1]

    # Sum of independent scores – provides partial credit
    comp_proxy = w_proxy * (score_x + score_y + score_vx + score_vy +
                            score_angle + score_contact)

    # ------------------------------------------------------------------
    # Component D – thrust penalty (efficiency cost, secondary)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)   # any engine use

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_proxy + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "success_proxy": comp_proxy,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components
```