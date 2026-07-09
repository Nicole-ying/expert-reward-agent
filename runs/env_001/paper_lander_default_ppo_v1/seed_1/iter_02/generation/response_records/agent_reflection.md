# Response Record

## 诊断分析

### 1. evidence
所有20条评估轨迹均在约68.5步时terminated（全部为早停、无truncated），score=-108；组件统计显示stability_penalty以-9.46的episode均值和78.9%的magnitude_share主导奖励，progress_reward仅+2.28，soft_landing_bonus几乎不触发（active_rate=0.4%）。

### 2. behavior_diagnosis
飞行器快速坠毁：episode平均68.5步即终止，stability_penalty每步都在扣分（active_rate=100%），累计惩罚远超progress_reward的正向引导，导致净奖励为负，agent无法学到有效的接近与减速行为。

### 3. signal_completeness
三个组件职责基本完备——progress提供接近引导、stability约束姿态与速度、soft_landing_bonus提供完成信号——但stability_penalty在尺度上压倒了progress_reward（|penalty/progress|≈4.15），使得progress引导失效。soft_landing_bonus条件合理但因agent从未到达着陆阶段而几乎不触发，不是当前瓶颈。

### 4. selected_level
**Level 1**：三个组件的数学形态与职责符号均合理，问题明确是stability_penalty系数过强导致尺度失衡（|penalty/progress|=4.15，远超0.5的经验触发线）。

### 5. selected_intervention
仅修改stability_penalty的四个系数，从`(0.1, 0.1, 0.1, 0.05)`降低为`(0.01, 0.01, 0.01, 0.005)`（统一缩小10倍），让penalty回归轻量约束角色，使progress_reward成为主导学习信号。

### 6. falsifiable_hypothesis
缩小stability_penalty后，|penalty/progress|应降至约0.4，progress_reward的正向引导不再被淹没，agent应能利用progress信号学到向着陆垫靠近的基本行为，episode length应增加、terminated原因可能从crash转向更晚的失败模式或出现部分truncated。

### 7. expected_next_round
stability_penalty的episode_sum_mean绝对值应下降至~1.0，progress_reward有望提升（因为存活更久、做出更多有效接近动作），soft_landing_bonus的active_rate可能略有上升，整体score应从-108显著回升，episode length应超过100步。

### 8. main_risk
惩罚过轻可能导致agent忽略姿态稳定，以高速、大角度姿态冲向着陆垫，虽然存活更久但无法完成软着陆——这将在下一轮表现为episode length上升但score仍然为负、soft_landing_bonus仍然不触发。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主学习信号：distance-based progress（只奖励接近，不惩罚远离） ----
    prev_distance = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_distance = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_distance - next_distance
    progress_reward = max(progress, 0.0) * 2.0

    # ---- 稳定/安全约束：轻量级惩罚，系数缩小10倍 ----
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    stability_penalty_value = (
        0.01 * abs(vx)
        + 0.01 * abs(vy)
        + 0.01 * abs(angle)
        + 0.005 * abs(angular_vel)
    )
    stability_penalty = -stability_penalty_value

    # ---- 任务完成近似信号：多条件组合的软着陆 bonus ----
    soft_landing_bonus = 0.0
    if (next_obs[6] == 1.0 and next_obs[7] == 1.0
            and abs(next_obs[0]) < 0.3
            and abs(next_obs[1]) < 0.1
            and abs(vx) < 0.3
            and abs(vy) < 0.3
            and abs(angle) < 0.2):
        soft_landing_bonus = 1.0

    # ---- 总奖励 ----
    total_reward = progress_reward + soft_landing_bonus + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components
```
