`evidence`：所有20个episode均terminated且平均长度仅76.45步（快速坠毁）；`landing_bonus`的active_rate仅1.0%，每episode约0.76步触发，属于极稀疏事件无法提供可学习梯度；`shaped_progress`虽占magnitude_share 71.7%且活跃99.9%，但只奖励向(0,0)移动，不区分软着陆与高速撞击；best（iter 5, score=120, len=597）表明该结构族有能力成功，但当前稀疏着陆信号导致agent学会快速接近目标后直接坠毁。

`behavior_diagnosis`：agent学会了向目标点快速逼近（shaped_progress正均值5.26），但缺乏减速信号，导致接近地面时以高速撞击而terminated，整体表现为“快速俯冲→坠毁”的失败模式。

`signal_completeness`：任务所需职责基本存在（进展引导、姿态稳定、着陆完成），但着陆信号`landing_bonus`的数学形态为`new_contact`门控的转移事件，active_rate仅1.0%，信用分配不可达；缺少近地面连续质量反馈，导致agent无法区分安全着陆与坠毁。

`selected_level`：Level 2 — `sparse_to_dense`变换，将`landing_bonus`从稀疏转移事件（new_contact门控）改为连续稠密信号（近地面质量评分），证据直接表明当前稀疏形态active_rate=1.0%无法提供有效引导，且所有episode均在~76步内坠毁。

`selected_intervention`：唯一改动目标组件为`landing_bonus`，将其从`new_contact * (加权和)`的稀疏转移事件改为`near_ground * (加权和)`的连续稠密近地面质量信号，重命名为`landing_quality`；移除了`new_contact`门控和`contact_before`依赖，改用`1/(1+5*|y|)`作为连续近地面因子。

`falsifiable_hypothesis`：近地面连续质量信号会在agent接近地面时持续给予低速度、直立、居中的正向反馈，使梯度从“高速撞击也得分”转为“减速接近才得分”，应引导agent学会在抵达目标区域前主动减速，从而降低坠毁率并提高实际着陆成功率。

`expected_next_round`：`landing_quality`的active_rate应从~1%显著上升至>30%；early_terminal比例应从3/20下降；score应改善（向best的120靠近）；episode_length应在不坠毁的前提下先延长（学会减速）后随策略成熟回落。

`main_risk`：连续近地面奖励可能使agent学会在目标区上方悬停（hovering）而非真正着陆——若episode_len显著变长但score未达best水平，说明悬停exploit发生，下一轮需叠加门控或接触确认。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 主学习信号：距离改善 × 姿态门控（保持不变）
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next

    angle_abs = abs(next_obs[4])
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)
    shaped_progress = 5.0 * progress_raw * gate

    # 2. 安全约束：角速度轻量惩罚（保持不变）
    ang_vel_penalty = -0.1 * (next_obs[5] ** 2)

    # 3. 着陆质量：连续稠密信号（sparse→dense，替代稀疏转移事件new_contact门控）
    height = next_obs[1]
    near_ground = 1.0 / (1.0 + 5.0 * abs(height))
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    low_speed = 1.0 / (1.0 + speed)
    upright = 1.0 / (1.0 + 3.0 * angle_abs)
    x_offset = abs(next_obs[0])
    centered = 1.0 / (1.0 + 3.0 * x_offset)

    landing_quality = 0.5 * near_ground * (2.0 * low_speed + 1.0 * upright + 1.0 * centered)

    total_reward = shaped_progress + ang_vel_penalty + landing_quality
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_quality": landing_quality
    }
    return (float(total_reward), components)
```