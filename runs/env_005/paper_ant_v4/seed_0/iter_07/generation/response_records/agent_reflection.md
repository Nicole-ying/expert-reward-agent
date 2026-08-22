# Response Record

1. evidence：上一轮门控结构导致episode满1000步但external_score跌至-172.4，forward_reward虽正值但反映前进速度极低（~0.007 m/s）、活跃度仅43.9%，且score远差于iter4最佳骨架的+26.07。
2. behavior_diagnosis：策略学会在满时长内极小幅度前移或间歇停滞，以极低速度存活、规避角速度和横向惩罚，未形成有效前进步态。
3. signal_completeness：已有前进、侧向、角速度信号及健康/直立门控，但前进激励强度不足以突破慢速局部最优；过往best骨架（iter4）用分离式惩罚更有效，证明软约束可行。
4. selected_level：Level 3 重建主骨架——当前骨架族（forward_reward + lat/ang penalty + 门控）连续两轮未刷新best、历史最佳仅26（远低于target 2000的25%），且已做Level 1尺度调整无效，满足重建条件。
5. selected_intervention：以iter4的组分框架为基础，将前进奖励从线性改为平方项（仅对正前向速度）以提供更强梯度，移除乘性健康/直立门控，改用温和的加性高度/直立惩罚维持存活，保留侧向与角速度惩罚。
6. falsifiable_hypothesis：平方化前进奖励使速度上升带来的边际收益递增，应能打破慢速徘徊的局部均衡；因此agent将学习产生更大且更持续的前向速度，外部得分会显著正向改善。
7. expected_next_round：下一轮score应转为正值并超出历史最佳26；forward_reward的episode_sum_mean和active_rate将大幅上升；episode仍以时间截断为主；角速度和侧向惩罚初期可能小幅上升。
8. main_risk：平方奖励可能驱动agent过度追求高速度而牺牲稳定性，导致偶发高度失控失败；需要软高度/直立惩罚约束，若约束不足可能训练不稳。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取状态
    body_z = next_obs[0]
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]

    # 前进奖励：仅对正前向速度使用平方，强化高速
    fwd_speed = max(0.0, body_x_vel)
    forward_reward = 15.0 * (fwd_speed ** 2)   # 目标速度~5时贡献~375/步

    # 软约束：高度惩罚（阈值0.3~0.9），仅在超出时生效
    height_low = max(0.0, 0.3 - body_z)
    height_high = max(0.0, body_z - 0.9)
    height_penalty = 20.0 * (height_low + height_high)

    # 直立惩罚：body_up_z = 1 - 2*(qx^2+qy^2)，低于0.5时惩罚
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_deficit = max(0.0, 0.5 - body_up_z)
    upright_penalty = 10.0 * upright_deficit

    # 横向速度惩罚
    lat_penalty = 2.0 * (body_y_vel ** 2)

    # 角速度惩罚
    ang_penalty = 0.5 * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    total_reward = forward_reward - height_penalty - upright_penalty - lat_penalty - ang_penalty

    components = {
        'forward_reward': forward_reward,
        'height_penalty': -height_penalty,
        'upright_penalty': -upright_penalty,
        'lat_penalty': -lat_penalty,
        'ang_penalty': -ang_penalty
    }
    return float(total_reward), components
```
