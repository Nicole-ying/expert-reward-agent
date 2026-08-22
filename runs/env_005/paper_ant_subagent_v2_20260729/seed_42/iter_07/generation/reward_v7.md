1. evidence: 上一轮 survival_bonus + gate‑forward_progress 得分 -585.8，len=663.7，forward_progress active_rate 仅 47.8%，per‑step original reward -1.06；历史最佳 iter 4 使用 forward_progress + lateral/roll/pitch hinge + descent_early_penalty 得分 5.5 是唯一正值，表明无门控前进 + 轻度本体约束有效，生存奖励有害。
2. behavior_diagnosis: agent 因 survival_bonus(无条件 +0.5/步) 倾向于站立或缓慢移动，而 gate‑forward_progress 有一半步数为零，整体正奖励不足抵消环境每步 -1.06 惩罚，导致缓慢退化。
3. signal_completeness: 前进激励、侧向漂移、滚转/俯仰角速度、身体高度均有可用连续信号，未缺失必备职责；当前缺口在于缺乏足够且持续的前进正信号，以及未利用高度越界抑制。
4. selected_level: Level 3 REBUILD，基于历史最优 iter 4 方向（无门控前进 + 轻量运动惩罚），移除生存奖励，引入高度边界惩罚，采用 dense_state_signal 正激励+ hinge 惩罚的主信号骨架。
5. selected_intervention: 重新构建奖励为 forward_reward(系数 2.0)、lateral_penalty(hinge,容忍 0.2)、roll/pitch_penalty(hinge,容忍 0.5)、height_low/height_high_penalty(hinge,阈 0.3/0.9)，完全移除 survival_bonus 及任何门控。
6. falsifiable_hypothesis: 无条件且较大前进奖励将迫使 agent 迈步以抵消环境惩罚，轻铰链约束在不压制探索的前提下抑制漂移与倾倒，整体应复现 iter 4 的举止并超过其得分。
7. expected_next_round: score > 0，len 回升至 600+，forward_reward active_rate ≈100%，lateral_penalty 与 roll/pitch_penalty 偶发触发，height_low_penalty 极少触发（agent 基本保持站立），terminated 比例下降。
8. main_risk: 若前进系数过大可能诱使 agent 为速度牺牲姿态导致早期摔倒；若高度 hinge 惩罚过重可能架空低矮探索区域，但当前权重尚可。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    forward_vel = next_obs[13]   # body x velocity
    lateral_vel = next_obs[14]   # body y velocity
    roll_vel    = next_obs[16]   # body roll velocity
    pitch_vel   = next_obs[17]   # body pitch velocity
    body_z      = next_obs[0]    # body height

    # 主前进奖励（线性，无门控）
    forward_reward = 2.0 * forward_vel

    # 侧向漂移惩罚 (hinge, 容忍 |lateral_vel| <= 0.2)
    lateral_penalty = -0.5 * max(0.0, abs(lateral_vel) - 0.2)

    # 滚转/俯仰角速度惩罚 (hinge, 容忍 |roll|,|pitch| <= 0.5)
    roll_penalty  = -0.2 * max(0.0, abs(roll_vel) - 0.5)
    pitch_penalty = -0.2 * max(0.0, abs(pitch_vel) - 0.5)

    # 身体高度边界惩罚 (hinge: 安全区域 [0.3, 0.9])
    height_low_penalty  = -1.0 * max(0.0, 0.3 - body_z)
    height_high_penalty = -1.0 * max(0.0, body_z - 0.9)

    total_reward = (forward_reward +
                    lateral_penalty + roll_penalty + pitch_penalty +
                    height_low_penalty + height_high_penalty)

    components = {
        'forward_reward': forward_reward,
        'lateral_penalty': lateral_penalty,
        'roll_penalty': roll_penalty,
        'pitch_penalty': pitch_penalty,
        'height_low_penalty': height_low_penalty,
        'height_high_penalty': height_high_penalty,
    }

    return float(total_reward), components
```