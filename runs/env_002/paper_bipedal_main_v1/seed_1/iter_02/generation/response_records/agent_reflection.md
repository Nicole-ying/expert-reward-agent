# Response Record

## 诊断与决策

1. **evidence**：第一轮训练，score=280.5、len=908、全部terminated；forward_reward episode_sum_mean=489.24（94.7% magnitude），stability_penalty=-27.38（5.3%），|penalty/progress|≈0.056远低于0.5的经验预警线；score_range=73-303表明episode间差异较大；任务描述明确要求“尽量降低能量消耗”但当前奖励完全没有动作能耗项。

2. **behavior_diagnosis**：机器人以中等速度（水平速度≈0.54）行走了约908步后终止，稳定性惩罚占比极小说明身体控制尚可但并非所有episode都稳定（得分最低仅73），整体处于一个可走路但不够优化、可能浪费能量的状态。

3. **signal_completeness**：前进引导（horizontal_velocity）和基本稳定性约束（倾斜/角速度/垂直速度）存在且可达，但任务明确要求的能量消耗职责完全缺失——这是一个必要但未被覆盖的信号。

4. **selected_level**：Level 2 —— 必要条件“能量消耗”信号缺失，触发`sparse_to_dense`反向场景的职责添加，且当前数学形态无明显塌缩或异常需要Level 1修复。

5. **selected_intervention**：新增唯一目标组件 `energy_penalty = -w_energy * sum(action_i²)`，w_energy=0.05；保持 forward_reward 和 stability_penalty 的系数与形态完全不变。

6. **falsifiable_hypothesis**：引入动作幅度代价后，策略将学会减少不必要的关节扭矩输出，节能步态通常也更稳定，预期下一轮中 energy_penalty 的 magnitude_share 在10-20%，episode 长度不降，得分方差收窄或 best 刷新。

7. **expected_next_round**：energy_penalty active_rate=100%，magnitude_share≈10-20%；score 短期可能因新增负项略微下降但 episode 长度应保持或增长；score_range 的下限应抬升（低分episode改善）。

8. **main_risk**：w_energy=0.05 若偏大可能过度抑制动作探索导致步态塌缩；若偏小则 energy_penalty 淹没在 forward_reward 中不起作用，需要下一轮根据 magnitude_share 验证。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant next observations
    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    horizontal_velocity = next_obs[2]
    vertical_velocity = next_obs[3]

    # Weights
    w_forward = 1.0
    w_angle = 1.0
    w_angvel = 0.2
    w_vert = 1.0
    w_energy = 0.05

    forward_reward = w_forward * horizontal_velocity
    stability_penalty = (
        -w_angle * abs(hull_angle)
        - w_angvel * abs(hull_angular_velocity)
        - w_vert * (vertical_velocity ** 2)
    )
    energy_penalty = -w_energy * (
        action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
    )

    total_reward = forward_reward + stability_penalty + energy_penalty
    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty
    }

    return float(total_reward), components
```
