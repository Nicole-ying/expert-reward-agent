# Response Record

`evidence`：所有20/20回合提前终止（len=68.5，score<-50），terminated=20/20；stability_penalty占magnitude_share 66.3%且active_rate 100%，说明速度/角度惩罚全程施加于仍需大动作移动的阶段；soft_landing_reward的episode_sum_mean仅0.35，远不足以引导减速着陆；progress_delta正向但无法抵消崩溃代价。

`behavior_diagnosis`：agent在到达目标前就飞出视口边界或撞击，全程被稳定性惩罚压制而不敢做大幅移动，同时缺乏边界危险预警，导致快速失败。

`signal_completeness`：缺少边界/碰撞避免信号和阶段感知——稳定性约束应在目标附近加强、在远处放权，当前全局施加的penalty在探索阶段起反作用。

`selected_level`：Level 2，触发条件为`global_to_local_gate`——约束在无关阶段（远离目标时）妨碍探索，证据是stability_penalty 100% active且agent无法生存。

`selected_intervention`：唯一修改stability_penalty，乘以距离门控因子`max(0.0, 1.0 - next_distance / 2.0)`，使速度/角度惩罚仅在距目标2单位内线性渐入，远处完全释放。

`falsifiable_hypothesis`：释放远处移动自由度后，agent应能更快靠近目标、延长存活步数；同时近目标区仍保留减速引导，不应出现终端速度过大撞击目标垫。

`expected_next_round`：episode_length应显著增长（>100步），terminated比例可能仍高但不再是全20/20极早终止；stability_penalty的magnitude_share和active_rate应下降（门控后触发范围缩小）；score应改善（负值得分减小）。

`main_risk`：gate_radius=2.0可能太小，若agent初始距离远超2.0，惩罚只在最后极短阶段生效，agent可能来不及减速就飞过/撞击目标垫，导致terminated模式从"边界飞出"变为"目标区碰撞"。下一轮若出现此模式，需增大gate_radius或补充分段减速信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D vehicle-like trajectory optimization:
    Reach and stabilize on the central target pad with minimal engine use.
    """
    # Previous observation
    px_prev = obs[0]
    py_prev = obs[1]
    prev_distance = (px_prev**2 + py_prev**2)**0.5

    # Next observation
    px = next_obs[0]
    py = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # Distance to target pad center
    next_distance = (px**2 + py**2)**0.5
    # Speed magnitude
    speed = (vx**2 + vy**2)**0.5

    # 1. Main learning signal: progress delta reward
    #    Positive when approaching target, negative when retreating.
    progress_delta = 3.0 * (prev_distance - next_distance)

    # 2. Stability constraint: light penalty on high speed, large angle, high angular velocity
    #    Now distance-gated: only active when agent is near the target pad (within ~2 units).
    #    This frees the agent to move decisively when far away.
    raw_stability = -0.1 * speed - 0.05 * abs(angle) - 0.05 * abs(ang_vel)
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_distance / gate_radius)
    stability_penalty = raw_stability * distance_gate

    # 3. Soft approaching proxy: reward getting close and slow simultaneously
    #    Acts as a smoothed "landing" surrogate without contact signals.
    sigma_dist = 0.2
    sigma_speed = 0.3
    nearness = 2.718281828 ** (-(next_distance**2) / (2.0 * sigma_dist**2))
    slowness = 2.718281828 ** (-(speed**2) / (2.0 * sigma_speed**2))
    soft_landing_reward = 1.0 * nearness * slowness

    # Combine components
    total_reward = progress_delta + stability_penalty + soft_landing_reward

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_reward': soft_landing_reward
    }

    return float(total_reward), components
```
