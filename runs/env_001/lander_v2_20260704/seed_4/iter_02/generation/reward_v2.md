## 诊断与干预

**evidence**: 最终策略得分-94.43，全部20回合在约68步非截断终止(early_terminal)。progress_reward总和均值+1.12、stability_penalty总和均值-0.95，二者接近抵消；landing_bonus总和均值仅0.005，active_rate=0.1%，实质上从未触发，无法提供任何任务完成引导。

**behavior_diagnosis**: 策略快速坠毁或出界——所有回合均在70步内终止，从未达成着陆条件。agent在progress引导下可能朝目标移动，但缺乏减速、调姿、接触的局部信号，导致高速撞击平台区域而失败。

**signal_completeness**: 缺少可达的任务完成信号。progress提供稠密方向引导，stability提供全局阻尼约束，但landing_bonus的二值稀疏形式(需同时满足6个硬条件)使完成信号对学习过程完全不可见——这是最关键的信号缺口。

**selected_level**: Level 2 — landing_bonus active_rate=0.1%直接满足sparse_to_dense触发条件，并非简单尺度问题。

**selected_intervention**: 将二进制稀疏`landing_bonus`替换为连续稠密`landing_proxy`。对每个着陆条件维度使用bounded函数`max(0, 1-|value|/D)`计算满意度，乘积组合为联合满足信号。放宽阈值(D_x=D_y=2.0, D_v=1.0, D_angle=0.5)确保信号可达，接触因子设置0.05基线防止乘积塌缩。系数2.0使完美着陆时每步贡献最大2.0，典型接近时(~0.06-0.7/步)与progress_reward尺度相当。

**falsifiable_hypothesis**: landing_proxy的active_rate应从0.1%显著上升(预期>10%)，agent应开始学习在目标附近减速和调姿，early_terminal应减少或episode长度增加，score应改善。

**expected_next_round**: landing_proxy的active_rate大幅上升(>10%)；early_terminal比例或严重度(score<-50)下降；episode_length可能先上升(存活更久)；score应有所改善但可能仍为负值。

**main_risk**: 乘积形式在agent远离目标时仍为零，初期梯度仍然稀疏；若D_x/D_y不够大覆盖agent的有效活动区域，landing_proxy可能仍然active_rate偏低。下一轮若active_rate仍<5%需加大阈值或改用加性组合。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 距离进展：每一步靠近目标的欧氏距离变化
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_obs - dist_next

    # 稳定性约束：抑制过度速度和姿态变化
    stability_penalty = (
        -0.01 * (abs(next_obs[2]) + abs(next_obs[3]))
        - 0.01 * abs(next_obs[4])
        - 0.01 * abs(next_obs[5])
    )

    # 稠密着陆代理信号：连续评估各着陆条件的满足度
    D_x = 2.0
    D_y = 2.0
    D_v = 1.0
    D_angle = 0.5

    x_sat = max(0.0, 1.0 - abs(next_obs[0]) / D_x)
    y_sat = max(0.0, 1.0 - abs(next_obs[1]) / D_y)
    vx_sat = max(0.0, 1.0 - abs(next_obs[2]) / D_v)
    vy_sat = max(0.0, 1.0 - abs(next_obs[3]) / D_v)
    angle_sat = max(0.0, 1.0 - abs(next_obs[4]) / D_angle)
    contact_factor = max(0.05, 0.5 * (next_obs[6] + next_obs[7]))

    landing_proxy = x_sat * y_sat * vx_sat * vy_sat * angle_sat * contact_factor
    landing_reward = 2.0 * landing_proxy

    total_reward = progress_reward + stability_penalty + landing_reward

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_proxy': landing_reward
    }

    return float(total_reward), components
```