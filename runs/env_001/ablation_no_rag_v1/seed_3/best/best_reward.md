1. `evidence`: score=93.87远低于target，episode长度878偏高，terminated/truncated=11/9；landing_proxy稀疏（active_rate 2.5%）但episode_sum_mean高，progress_reward稠密但占比极低（magnitude_share 3.3%），velocity_penalty持续施加但数值不大。  
2. `behavior_diagnosis`: agent在大部分episode徘徊或缓慢接近，成功着陆的episode占比不高，且着陆后可能触发持续landing_proxy导致episode被截断；关键的靠近驱动progress_reward过于微弱，无法有效引导快速接近。  
3. `signal_completeness`: 必要职责基本完备：稠密进程引导(progress_reward)、稳定性约束(velocity/attitude)、着陆代理(landing_proxy)，但尺度严重失衡导致引导失效。  
4. `selected_level`: Level 1，因为组件符号与数学形态合理，仅存在明显的尺度不对称，progress_reward相对其他组件过弱。  
5. `selected_intervention`: 向progress_reward引入系数w_progress=5.0，放大其贡献，其他组件不变。  
6. `falsifiable_hypothesis`: 增大稠密进程奖励将使agent更快向目标移动，提升接近效率与着陆成功率，从而拉高平均得分并缩短episode长度。  
7. `expected_next_round`: progress_reward的episode_sum_mean约增至6.8，score应显著提升，episode length可能下降，velocity_penalty可能因运动速度增加而微幅上升。  
8. `main_risk`: 过强的进程奖励可能让agent忽略减速，增加高速撞击失败的风险，但现有的velocity_penalty和landing_proxy仍有一定约束力。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next state variables
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distance to landing pad center (Euclidean)
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5

    # 1. Main learning signal: progress toward the pad (scaled up)
    w_progress = 5.0
    progress_reward = w_progress * (dist - next_dist)

    # 2. Stability penalty: penalize high velocities
    w_vel = 0.1
    velocity_penalty = -w_vel * (vx**2 + vy**2)

    # 3. Attitude penalty: penalize non-zero body angle (want upright)
    w_angle = 0.01
    attitude_penalty = -w_angle * abs(angle)

    # 4. Soft landing proxy: encourage gentle touchdown with both legs
    w_landing = 2.0
    alpha = 20.0   # sharpness for position proximity
    beta = 5.0     # sharpness for low speed
    contact_factor = left_contact * right_contact  # 1 if both legs touch, else 0
    position_proximity = 2.718281828 ** (-alpha * (next_x**2 + next_y**2))
    speed_term = 2.718281828 ** (-beta * (vx**2 + vy**2))
    landing_proxy = w_landing * contact_factor * position_proximity * speed_term

    total_reward = progress_reward + velocity_penalty + attitude_penalty + landing_proxy

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "attitude_penalty": attitude_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components
```