## 诊断与干预

`evidence`：当前策略与历史最佳完全相同，score 无变化（2663），全部 episode 跑满 1000 步被截断，无提前失败；forward_stability_reward 以 99.1% magnitude share 占据绝对主导，stability_penalty 仅 -13.4 几乎可忽略；奖励结构中完全缺失垂直方向运动信号。

`behavior_diagnosis`：策略已学会保持直立并产生持续前向速度以在 1000 步内存活，但没有被激励执行跳跃所必需的垂直振荡运动——agent 极可能在用低能耗的"贴地滑步/拖行"策略而非真正的跳跃步态来获取 forward 奖励，这限制了前进速度和任务表现的上限。

`signal_completeness`：任务核心是"连续向前跳跃"，当前奖励仅覆盖了"向前"和"直立"，完全缺失"跳跃"所必需的垂直运动/离地动态信号——这是一个必要职责的缺口，且该信号（vertical_velocity）在 obs 中直接可用、可达。

`selected_level`：Level 2（必要信号缺失）。跳跃任务的垂直振荡动态完全没有被奖励函数表达，仅靠 forward_vel * upright_factor 无法区分跳跃步态与贴地滑行，需要在数学结构上增加一个垂直活动组件来弥补信号缺口。

`selected_intervention`：新增 `vertical_activity` 组件——对垂直速度的绝对值进行奖励，并用同一 upright_factor 门控，确保只在直立姿态下鼓励跳跃的升降动态。系数设为 0.2，使其成为辅助信号而不会压倒主前进目标。

`falsifiable_hypothesis`：增加垂直活动信号后，agent 将被迫探索真正的跳跃步态（因为贴地滑行时 vertical_velocity 近似为零，无法获得该奖励），从而在保持直立的同时解锁更高的前进速度和更自然的跳跃周期，最终提升外部 score。

`expected_next_round`：vertical_activity 的 active_rate 应接近 100%，magnitude_share 预计在 5-20% 之间；若跳跃步态被解锁，forward_stability_reward 的 episode_sum_mean 应上升（更快的真实跳跃带来更高前向速度），外部 score 应刷新 best；episode_length 保持 1000（存活能力已具备）。

`main_risk`：vertical_activity 可能鼓励原地上下跳动而不前进（垂直活动与前进脱耦），但由于 forward_stability_reward 仍是主导项（预计占 80%+），且 upright_factor 对倾斜会衰减两个组件，该风险可控；若下一轮出现 forward 下降但 vertical 上升，说明需要进一步耦合两个信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for hopping locomotion: forward progress + vertical activity + stability.
    Components:
        forward_stability_reward – forward velocity weighted by torso uprightness
        vertical_activity         – absolute vertical velocity weighted by uprightness (hopping signal)
        stability_penalty         – penalty for torso tilt and angular velocity
    """
    # --- extract relevant signals from next_obs ---
    torso_angle = next_obs[1]          # rad, 0 = upright
    forward_vel = next_obs[5]          # positive = forward
    vertical_vel = next_obs[6]         # vertical velocity (hopping oscillation)
    torso_ang_vel = next_obs[7]        # rad/s

    # --- shared upright gate ---
    temp = 0.3
    upright_factor = 2.718281828 ** (-abs(torso_angle) / temp)

    # --- forward progress with upright gating ---
    forward_stability_reward = forward_vel * upright_factor

    # --- vertical activity: encourage the up-down dynamics of hopping ---
    w_vert = 0.2
    vertical_activity = w_vert * abs(vertical_vel) * upright_factor

    # --- light stability penalty ---
    w_angle = 0.1
    w_ang_vel = 0.01
    stability_penalty = -w_angle * (torso_angle ** 2) - w_ang_vel * (torso_ang_vel ** 2)

    # --- total reward ---
    total_reward = forward_stability_reward + vertical_activity + stability_penalty

    components = {
        "forward_stability_reward": forward_stability_reward,
        "vertical_activity": vertical_activity,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```