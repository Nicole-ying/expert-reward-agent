# Response Record

1. `evidence`：上一轮得分-24.96，全部episode达到1000步后被截断（truncated=20/20），landing_reward占signed share 99.5%且episode sum mean高达686.5，但外部score仍为负且未触发任何终止，说明大量奖励来自悬停而非着陆。

2. `behavior_diagnosis`：agent学会在高空中维持小距离、小角度、小速度的“好状态”，持续获取landing_reward的连续状态奖励，从而放弃下降和接触，从不触发着陆终止。

3. `signal_completeness`：缺少成功终端信号和防止状态停留的改善机制；landing_reward为纯state奖励，允许通过占据好状态无限累积奖励，而未驱动朝向最终着陆点移动。

4. `selected_level`：Level 2 — state→improvement结构变换，将landing_reward从连续状态奖励改为基于潜在函数差分的改善奖励。

5. `selected_intervention`：删去original landing_reward，替换为距离、角度、速度的潜在差分奖励、接触增益奖励以及一次性成功着陆奖励。

6. `falsifiable_hypothesis`：取消悬停奖励后，agent将被迫向更低势能（更靠近平台、更竖直、更慢）移动以获取正向差分，从而触发接触与成功条件，最终提升外部score。

7. `expected_next_round`：episode length可能缩短（出现成功或坠毁终止），terminated比例上升，score改善（趋向正值），landing组件均值与份额大幅下降，contact_bonus与success_bonus出现。

8. `main_risk`：成功条件阈值过于严格导致稀疏奖励难以探索，agent可能因下降过快坠毁而得分更差；差分尺度若偏小可能推进力不足。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, n_angle, n_ang_vel, n_left, n_right = next_obs

    # ---------- 1. Distance progress (keep) ----------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    delta_dist = dist - next_dist
    progress_reward = 2.0 * delta_dist

    # ---------- 2. Attitude safety penalty (keep) ----------
    angle_err = abs(n_angle)
    ang_vel_abs = abs(n_ang_vel)
    attitude_penalty = -0.5 * (angle_err**2 + (0.5 * ang_vel_abs)**2)

    # ---------- 3. Landing potential difference (replaces state-based landing_reward) ----------
    # Potential function: lower is better (closer, more upright, slower)
    speed = (vx**2 + vy**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5
    angle_err_prev = abs(angle)
    angle_err_next = abs(n_angle)

    pot_prev = - (5.0 * dist + 10.0 * angle_err_prev + 5.0 * speed)
    pot_next = - (5.0 * next_dist + 10.0 * angle_err_next + 5.0 * next_speed)
    landing_potential_diff = pot_next - pot_prev

    # Contact gain: reward for newly establishing leg contacts
    contact_gain = (n_left + n_right) - (left_contact + right_contact)   # 0, 1, or 2
    contact_bonus = 10.0 * contact_gain

    # Success bonus: approximately landed
    success = (
        n_left == 1.0 and n_right == 1.0 and
        abs(nvx) < 0.2 and abs(nvy) < 0.2 and
        abs(n_angle) < 0.2 and
        abs(nx) < 0.5 and abs(ny) < 0.5
    )
    success_bonus = 100.0 if success else 0.0

    landing_reward = landing_potential_diff + contact_bonus + success_bonus

    # ---------- Aggregate ----------
    total_reward = progress_reward + attitude_penalty + landing_reward

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": attitude_penalty,
        "landing_potential_diff": landing_potential_diff,
        "contact_bonus": contact_bonus,
        "success_bonus": success_bonus
    }
    return float(total_reward), components
```
