# Response Record

1. `evidence`：第七轮得分-133.27，contact_bonus 占 magnitude_share 94.6% 但 active_rate 仅 1.1%，progress、fuel_penalty 等被淹没，且预判连续三轮❌，表明当前方向错误。  
2. `behavior_diagnosis`：策略只在双腿同时接触的极少数步获得巨额稀疏奖励，缺失下降过程的连续引导，导致训练出的行为过程低效、姿态波动大，外部得分极低。  
3. `signal_completeness`：任务需要下降进程引导、姿态稳定和成功接触奖励，现有组件架构可覆盖，但 contact_bonus 的极度稀疏形态破坏了引导完整性。  
4. `selected_level`：Level 2，因 contact_bonus 数学形式（仅接触时大额指数奖励）直接导致行为崩塌，无需更换整体骨架，改换组件形态即可。  
5. `selected_intervention`：唯一干预目标为 contact_bonus，将其从“仅双腿接触给大额奖励”改为“近地连续引导（y<1.0 给予下降姿态奖励）+ 接触时额外奖励”，并将期望下降速度从 -0.5 调整为 -1.0 以加快着陆。  
6. `falsifiable_hypothesis`：密集的近地引导信号使 agent 每步获得下降与姿态反馈，接触奖励保底精确定位，从而稳定学习有效下降策略，提升外部得分、缩短 episode、减少燃料浪费。  
7. `expected_next_round`：score 应大幅回升至正区间（>0），len 降至 600‑800，contact_bonus active_rate 升至 >80%，terminated 次数可能略降但仍维持多数成功，fuel_penalty 绝对值可能略增但总奖励更高。  
8. `main_risk`：加快期望下降速度可能导致着陆冲击力增大，部分 episode 触发 crash 终止，若 weight 未配合则可能引入新的失败模式，需观察 crash 率。  

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

    # ---------- contact_bonus: near-ground continuous guidance + contact lump sum ----------
    descent_target_vy = -1.0   # 加快下降的期望速度
    vy_error = vy_next - descent_target_vy
    soft_factor = (2.718281828 ** (-(vy_error**2) / 0.1)) * \
                  (2.718281828 ** (-(vx_next**2) / 0.1)) * \
                  (2.718281828 ** (-(angle_next**2) / 0.05)) * \
                  (2.718281828 ** (-(angvel_next**2) / 0.05))

    contact_bonus = 0.0
    if y_next < 1.0:
        contact_bonus = 5.0 * soft_factor   # 近地连续引导

    if left_contact_next > 0.5 and right_contact_next > 0.5:
        contact_bonus += 30.0 * soft_factor # 接触时额外稳定性奖励

    # ---------- fuel penalty ----------
    fuel_penalty = -0.05 if action != 0 else 0.0

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
