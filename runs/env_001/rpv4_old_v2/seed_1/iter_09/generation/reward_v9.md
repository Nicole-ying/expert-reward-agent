1. `evidence`：当前20/20终止、全部坠毁，contact_bonus主导（per‑step 1.076 vs progress 0.026），近地层奖励给出大量无接触引导；历史最佳iter3（91.82分）使用更低的期望下降速度（-0.5）并包含y<1奖励，而本轮类似结构却失败，说明近地层奖励提供了错误激励。

2. `behavior_diagnosis`：策略被contact_bonus的近地层无条件奖励误导，学会了快速撞地向地面获取奖励，而不是学习低速软着陆并双腿触地，因此全数坠毁。

3. `signal_completeness`：现有奖励具备下降引导（progress）、软着陆奖励（contact_bonus基于姿态/速度）、燃料惩罚和近地姿态惩罚；但近地层无条件奖励破坏了软着陆目标的纯净性，移除它后信号依旧完备且可达。

4. `selected_level`：Level 2——修改组件的数学形式。虽然连续多轮失败，但历史最佳证明骨架方向可行，无需重建；只需修正contact_bonus中传达错误激励的`y_next<1.0`分支。

5. `selected_intervention`：唯一目标组件`contact_bonus`，移除“`if y_next < 1.0` 给予 5.0 * soft_factor”的分支，仅保留双腿接触时给予 30.0 * soft_factor 的奖励，其余参数与历史最佳代码一致（`descent_target_vy = -0.5`）。

6. `falsifiable_hypothesis`：去除近地无接触奖励后，策略不再因接近地面即得高分而提前坠毁，必须完成双腿稳定接触才能获得大额奖励，因此应转向学习精确软着陆，提高成功着陆并减少坠毁。

7. `expected_next_round`：`contact_bonus`的episode_sum_mean下降（因不再有近地持续性奖励），active_rate可能上升（更多轨迹最终触地）；总score应提升，len增长，terminated减少、truncated增加。

8. `main_risk`：奖励变得稀疏可能造成探索困难，策略可能长时间悬停而无法发现双腿接触的正确姿态，导致truncation比例过高或进步缓慢；后续可能需要引入中等难度引导信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- progress reward ----------
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5
    progress_delta = dist_obs - dist_next

    speed_sq = vx_next**2 + vy_next**2
    gate_speed = 1.0 / (1.0 + 0.1 * speed_sq)
    gate_angle = 1.0 / (1.0 + 0.5 * abs(angle_next))
    gate_health = gate_speed * gate_angle
    progress_reward = 5.0 * progress_delta * gate_health

    # ---------- contact_bonus: only when both legs touch ----------
    descent_target_vy = -0.5
    vy_error = vy_next - descent_target_vy
    soft_factor = (2.718281828 ** (-(vy_error**2) / 0.1)) * \
                  (2.718281828 ** (-(vx_next**2) / 0.1)) * \
                  (2.718281828 ** (-(angle_next**2) / 0.05)) * \
                  (2.718281828 ** (-(angvel_next**2) / 0.05))

    contact_bonus = 0.0
    if left_contact_next > 0.5 and right_contact_next > 0.5:
        contact_bonus = 30.0 * soft_factor

    # ---------- fuel penalty ----------
    fuel_cost = 0.05
    fuel_penalty = -fuel_cost if action != 0 else 0.0

    # ---------- near-ground orientation penalty ----------
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