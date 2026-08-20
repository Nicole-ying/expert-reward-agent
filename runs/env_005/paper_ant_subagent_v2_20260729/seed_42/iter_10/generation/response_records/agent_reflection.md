# Response Record

1. `evidence`：最终得分-375.55，终止率50%，ep_len=662，forward_reward episode_sum_mean=2742（magnitude_share 90.5%），但 upright_penalty（signed_share -7.6%，active_rate 50.7%）是最大负贡献；roll_vel/ascent/pitch 惩罚几乎 dead；历史 best 仅 5.5（iter 4），随后连续 5 轮无改善，且本轮 score 为负——说明当前“前进+一堆独立惩罚”骨架已经失效。
2. `behavior_diagnosis`：策略通过身体倾斜换取高前进速度，但直接导致 uprightness 恶化、频繁触发终止（高度越界/过度倾斜），呈现典型的“先冲后死”模式——健康状态下降时仍追逐前进奖励，未受到有效抑制。
3. `signal_completeness`：身体高度、四元数、前进速度等必要信号均可用，当前缺的是将健康状态直接乘到主进展奖励上的 gate 机制，而非继续叠加独立惩罚；gate 能够更早干预 agent 的行为。
4. `selected_level`：Level 3 重建，触发条件：同一“前进+独立惩罚”骨架族连续 5 轮未刷新 best（iter 5‑9 全部负分），且历史最高仅 5.5，远未达到目标。
5. `selected_intervention`：完全替换为 soft_health_gate 骨架——主奖励为 `forward_vel * height_gate * uprightness_gate`，height_gate 在 [0.35,0.9] 内恒为 1，向两侧线性衰减至 0；uprightness_gate 取 `max(0, body_up_z)`；辅以轻量的 lateral/角速度/关节速度/动作惩罚。
6. `falsifiable_hypothesis`：如果 gate 机制能将前进奖励的成功与身体稳定状态绑定，那么 agent 必须在保持高度和直立的前提下最大化速度，预测下一轮的 episode 平均总分应转为正值（>0），且 uprightness_gate 的均值接近 1，terminated 比例下降，ep_len 至少维持 700 以上。
7. `expected_next_round`：score 应变为正数（>0），len≥700，terminated 占比 <40%；forward_reward 的 signed_share 仍显著，height_gate 和 uprightness_gate 的 active_rate 接近 100%，而 lateral/角速度等惩罚值保持较小。
8. `main_risk`：gate 衰减区间可能过窄或 training_progress 早期身体不稳定时 gate 频繁接近零，导致主奖励信号几乎消失，agent 学不会前进而停滞或更早终止，表现为 len 进一步缩短且 score 仍为负。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Key signals from next_obs
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    
    # uprightness: body_up_z = 1 - 2*(qx^2 + qy^2), ∈[-1,1] (1 = perfectly upright)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    
    forward_vel = next_obs[13]
    lateral_vel = next_obs[14]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    
    # ---- Soft health gates ----
    # Height gate: safe zone [0.35, 0.9], danger margins [0.2, 1.0]
    lower_safe = 0.35
    lower_danger = 0.2
    upper_safe = 0.9
    upper_danger = 1.0
    gate_lower = min(1.0, max(0.0, (body_z - lower_danger) / (lower_safe - lower_danger)))
    gate_upper = min(1.0, max(0.0, (upper_danger - body_z) / (upper_danger - upper_safe)))
    height_gate = min(gate_lower, gate_upper)
    
    # Uprightness gate: use body_up_z directly, clamp negatives to zero
    uprightness_gate = max(0.0, body_up_z)
    
    # ---- Main progress reward ----
    # Forward velocity scaled by both health gates
    forward_reward = 2.0 * forward_vel * height_gate * uprightness_gate
    
    # ---- Light constraints ----
    lateral_penalty = -0.1 * max(0.0, abs(lateral_vel) - 0.5)
    roll_vel_penalty = -0.005 * roll_vel**2
    pitch_vel_penalty = -0.005 * pitch_vel**2
    
    # Joint velocity smoothing (very light)
    joint_vel_penalty = -0.0002 * sum(next_obs[19:27]**2)   # indices 19..26
    
    # Action magnitude (energy cost, tiny)
    action_penalty = -0.0005 * sum(a**2 for a in action)
    
    total = (forward_reward +
             lateral_penalty +
             roll_vel_penalty +
             pitch_vel_penalty +
             joint_vel_penalty +
             action_penalty)
    
    components = {
        'forward_reward': forward_reward,
        'height_gate': height_gate,
        'uprightness_gate': uprightness_gate,
        'lateral_penalty': lateral_penalty,
        'roll_vel_penalty': roll_vel_penalty,
        'pitch_vel_penalty': pitch_vel_penalty,
        'joint_vel_penalty': joint_vel_penalty,
        'action_penalty': action_penalty
    }
    
    return float(total), components
```
