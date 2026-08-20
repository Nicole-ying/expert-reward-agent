# Response Record

1. `evidence`：score=-415.21，len=299，terminated=15/20；forward_reward magnitude_share 72% 但总值仅 74.7，其余五项惩罚均为负；height_low_penalty active_rate=74.7%，height_high_penalty 仅 1.6%；quat（obs[1:5]）和 body_z_velocity 均未使用；终止模式以 failed termination 为主且长度短，指向频繁因倾斜/高度过低导致的摔倒。

2. `behavior_diagnosis`：策略试图前进但频繁失去平衡导致提前终止，forward_reward 虽有正值但稳定性不足，身体倾斜过大引发摔倒，惩罚组件只约束角速度而缺失对躯体直立程度的直接反馈，导致主信号无法引导稳定行走。

3. `signal_completeness`：缺少对 body uprightness 的连续反馈——四元数可计算 body_up_z 作为身体垂直方向的投影，该信号可用于构建 gate，防止 agent 在严重倾斜时仍获得高速奔跑奖励从而陷入不稳定状态；整体职责存在缺口。

4. `selected_level`：Level 2（结构变换）——累积记录显示连续 3 轮未刷新 best，当前“纯线性前进+多项独立惩罚”骨架持续失效；需要在前进奖励中引入 uprightness gate 作为局部反馈，将二值安全条件变为连续的 bounded factor，符合“sparse→dense”模式。

5. `selected_intervention`：将 `forward_reward` 组件从 `2.0 * forward_vel` 修改为 `2.0 * forward_vel * uprightness_gate`，其中 `uprightness_gate` 由四元数计算并通过 hinge 函数映射为 [0,1] 连续值；其余惩罚组件保持不变，仅此一个组件变更。

6. `falsifiable_hypothesis`：增加直立性门控后，agent 只有在保持身体接近直立时才能获得前进奖励，这将迫使它学习在稳定姿态下前进，从而降低以大幅倾斜换取速度的行为，使 terminated 率下降、episode 长度回升，最终提升外部 score。下一轮可通过 terminated 比例和 len 检验。

7. `expected_next_round`：terminated 比例应从 15/20 下降，len 应回升至 500-800 区间；forward_reward 的 episode_sum_mean 可能略降至 30-50，但总 score 将上升至 0~100；height_low_penalty active_rate 可能下降至 40% 以下；uprightness_gate active_rate 接近 100%。

8. `main_risk`：gate 阈值设置过严导致 agent 在早期探索中完全无法获得正奖励，陷入零前进的停滞状态；若发生，下一轮需调整 gate 的下限或提高 forward_reward 系数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    forward_vel = next_obs[13]   # body x velocity
    lateral_vel = next_obs[14]   # body y velocity
    roll_vel    = next_obs[16]   # body roll velocity
    pitch_vel   = next_obs[17]   # body pitch velocity
    body_z      = next_obs[0]    # body height

    # 计算身体直立度：body_up_z = 1 - 2*(qx^2 + qy^2)
    qx = next_obs[2]
    qy = next_obs[3]
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)  # 范围[-1,1]，1为完全直立

    # 直立性门控：当 body_up_z 低于 0.3 时门控归零，线性过渡到 1.0 时门控为 1
    uprightness_gate = max(0.0, (body_up_z - 0.3) / 0.7)

    # 主前进奖励（引入直立性门控）
    forward_reward = 2.0 * forward_vel * uprightness_gate

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
