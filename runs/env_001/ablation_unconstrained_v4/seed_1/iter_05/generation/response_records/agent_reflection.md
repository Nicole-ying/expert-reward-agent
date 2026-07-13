# Response Record

**分析与修改理由**

当前agent能稳定完成着陆（terminated=20/20，无truncate），最终score主要来自少帧数的`landing_bonus`（触发率仅2.8%但累积1655），说明成功着陆时的乘积奖励已经有效。然而episode长度高达647步，说明agent在到达目标垫前花费了大量时间在空中浮游或缓慢下降，缺乏快速接近的动力。证据是`distance_progress`的sum_mean只有2.4（占0.1%），且active_rate达88.5%表明几乎每步都在计算progress，但每步的progress非常小（约0.0037），奖励信号极弱。任务要求“以尽可能快的速度到达并稳定降落”，当前奖励未能驱动快速趋向目标。另外`contact_reward`贡献较大（9.4%）可能鼓励腿部接触但并未缩短时间，没有燃料惩罚也允许agent随意使用主引擎维持高度。

因此，干预方向：
1. **强化progress信号**：不再只奖励正向progress，而是改为双向shaping（靠近奖、远离惩），并提高系数，促使agent持续向目标移动。
2. **添加生存惩罚**：每步给予极小负奖励，直接惩罚拖长时间。
3. **添加主引擎燃料惩罚**：对`action=2`施加小额惩罚，鼓励利用惯性滑翔，既省燃料又可能因减少不必要的悬浮而加快着陆。
landing_bonus结构已证明有效，保持不变。

根据历史记忆，当前代码已是best，在此基础上进行最小可证伪修改。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # extract current and next state
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # distances from target
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # progress (positive if getting closer, negative if drifting away)
    progress = prev_dist - next_dist
    c_progress = 5.0 * progress   # stronger bidirectional shaping

    # contact reward
    contact_count = n_left_contact + n_right_contact
    c_contact = 2.0 * contact_count

    # posture penalties
    c_angle = -0.05 * (n_angle**2)
    c_angvel = -0.01 * (n_angvel**2)

    # per-step alive penalty to encourage faster completion
    c_alive = -0.005

    # fuel penalty for using main engine (action 2)
    fuel_penalty = -0.02 if (action == 2) else 0.0

    # soft-landing bonus (unchanged, only when both legs contact)
    two_legs = (n_left_contact > 0.5 and n_right_contact > 0.5)
    if two_legs:
        dist = next_dist
        speed_mag = (nvx**2 + nvy**2) ** 0.5
        angle_mag = abs(n_angle)
        angvel_mag = abs(n_angvel)

        score_dist = max(0.0, 1.0 - dist / 0.3)
        score_speed = max(0.0, 1.0 - speed_mag / 0.3)
        score_angle = max(0.0, 1.0 - angle_mag / 0.2)
        score_angvel = max(0.0, 1.0 - angvel_mag / 0.3)

        c_landing = 200.0 * score_dist * score_speed * score_angle * score_angvel
    else:
        c_landing = 0.0

    total_reward = (c_progress + c_contact + c_angle + c_angvel +
                    c_landing + c_alive + fuel_penalty)

    components = {
        'distance_progress': c_progress,
        'contact_reward': c_contact,
        'angle_penalty': c_angle,
        'angvel_penalty': c_angvel,
        'landing_bonus': c_landing,
        'alive_penalty': c_alive,
        'main_engine_penalty': fuel_penalty,
    }
    return (float(total_reward), components)
```
