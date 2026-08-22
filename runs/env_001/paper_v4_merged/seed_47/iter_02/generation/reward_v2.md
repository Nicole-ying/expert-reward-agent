1. **evidence**：上一轮 score=-36.03，所有 episode truncated，success_reward 的 active_rate=0% 且 episode_sum_mean=0.0，唯一从未触发的组件；progress_reward 和 attitude_penalty 持续活跃但无法引导任务完成。
2. **behavior_diagnosis**：agent 学会了缓慢接近目标并维持姿态，但始终未能满足软着陆判据，导致每局耗满 1000 步无成功终止。
3. **signal_completeness**：现有 progress 和姿态约束可覆盖接近与姿态，但因 success_reward 从未激活，缺失从“接近”到“着陆”的连续引导信号。
4. **selected_level**：Level 2 结构变换，触发条件：active_rate=0% 的僵尸组件必须替换为可连续获利的信号，当前乘积式 proxy 塌缩。
5. **selected_intervention**：将 `success_reward` 组件替换为 `landing_reward`，由四个独立因子的平均值构成，消除乘积塌缩并使每个因子均可独立提供梯度。
6. **falsifiable_hypothesis**：新 `landing_reward` 的 active_rate 将显著 >0，每 episode 累计正值，引导 agent 在接近目标时同步改善姿态、减速和接触，从而提升整体 score。
7. **expected_next_round**：`landing_reward` 的 episode_sum_mean 应为正（约 0.5~2.0），active_rate≈100%，总 score 上升至少 5 点，可能仍会被 attitude_penalty 拉负，但负值减少。
8. **main_risk**：agent 可能过早追求直立和静止以获取 `landing_reward`，导致 progress 减慢甚至停滞，但 progress 的系数较大仍可维持前进动力。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, n_angle, n_ang_vel, n_left, n_right = next_obs

    # ---------- 1. Main progress: improvement in Euclidean distance to landing pad ----------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    delta_dist = dist - next_dist                # positive when getting closer
    progress_reward = 2.0 * delta_dist

    # ---------- 2. Attitude safety constraint ----------
    angle_err = abs(n_angle)
    ang_vel_abs = abs(n_ang_vel)
    attitude_penalty = -0.5 * (angle_err**2 + (0.5 * ang_vel_abs)**2)

    # ---------- 3. Landing approach reward (continuous multi-factor, replaces dead success_reward) ----------
    prox = max(0.0, 1.0 - next_dist / 5.0)
    upright = max(0.0, 1.0 - angle_err / 0.5)
    speed = (nvx**2 + nvy**2) ** 0.5
    stationary = max(0.0, 1.0 - speed / 1.0)
    contact = (n_left + n_right) / 2.0
    landing_reward = 1.0 * (prox + upright + stationary + contact) / 4.0

    # ---------- Aggregate ----------
    total_reward = progress_reward + attitude_penalty + landing_reward

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": attitude_penalty,
        "landing_reward": landing_reward
    }
    return float(total_reward), components
```