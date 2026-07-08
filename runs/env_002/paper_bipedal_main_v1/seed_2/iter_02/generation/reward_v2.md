`evidence`：上一轮首次迭代，得分272.07，20/20全部terminated（非truncated），episode长度~1285步且得分极窄[268,275]，forward_reward占magnitude 88.7%、stability_penalty占11.3%，ratio约0.13；绝对垂直速度和躯干倾角都很小，策略已稳定行走但外部得分与内部reward sum（~440）差距明显，推测外部有独立评分或环境已走完全程。

`behavior_diagnosis`：策略以约0.39单位/步的中等速度稳定前进了约1285步，躯干接近直立（倾角~0.04rad），垂直振荡极小（~0.008单位/步），所有episode都一致终止于地形终点或摔倒点，表现稳定但可能处于保守步态的局部最优。

`signal_completeness`：任务明确要求"节省能量"，当前奖励却缺少任何能耗反馈；前进速度作为进展代理基本合理，稳定性约束轻且有效；缺失的能耗信号是当前最值得补充的职责。

`selected_level`：Level 2 —— 必要信号缺失（任务明确要求节能但无对应奖励），属于sparse_to_dense的反向补全：为已有职责集合增加缺失的能耗组件。

`selected_intervention`：新增energy_penalty组件 = -0.05 * sum(action_i^2)，惩罚四个关节扭矩平方和，其他两个组件完全不变。

`falsifiable_hypothesis`：增加适度动作能耗惩罚应促使策略使用更小、更平滑的关节扭矩来维持前进速度，利用动量而非蛮力，从而产生更高效的自然步态；如果外部评分隐含效率指标，外部得分应上升；即使外部得分不变，agent的步态质量应改善。

`expected_next_round`：energy_penalty的active_rate=100%、magnitude_share预计在5-15%之间；forward_reward应保持或略升（更高效步态可能解放速度）；stability_penalty可能小幅下降；外部score变化取决于环境评分函数，但不应明显恶化。

`main_risk`：若w_energy=0.05偏大，可能过度抑制动作幅度导致前进速度下降；若偏小则无效果，下一轮可能需要根据energy_penalty magnitude调整系数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D bipedal locomotion on rough terrain.
    Drives forward progress, maintains stable posture, and encourages energy-efficient motion.
    """
    # ---------- forward progress component ----------
    forward_velocity = next_obs[2]
    w_fwd = 1.0
    forward_reward = w_fwd * forward_velocity

    # ---------- stability penalty component ----------
    hull_angle = next_obs[0]
    vertical_velocity = next_obs[3]
    w_angle = 0.5
    w_vertical = 0.1
    stability_penalty = -w_angle * abs(hull_angle) - w_vertical * abs(vertical_velocity)

    # ---------- energy efficiency component ----------
    w_energy = 0.05
    energy_penalty = -w_energy * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ---------- total reward ----------
    total_reward = forward_reward + stability_penalty + energy_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty
    }

    return float(total_reward), components
```