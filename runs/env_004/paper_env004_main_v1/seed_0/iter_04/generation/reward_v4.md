1. `evidence`：Current score 1338 vs best 2663，加入 `vertical_activity = 0.2 * abs(vertical_vel) * upright_factor` 后 episode length 从 1000 骤降至 444，19/20 提前终止；`stability_penalty` 近乎为零（-0.55），说明 agent 不是因为倾斜惩罚而死，而是 vertical 信号诱导了导致越界的剧烈垂直运动。

2. `behavior_diagnosis`：agent 在 `abs(vertical_vel)` 的激励下产生了过度垂直运动——向上可能超出健康高度、向下可能冲击倒地，最终触发 height/angle 越界终止；同时 `abs()` 对上升和下降对称奖励，下降阶段是被动坠落而非主动控制，奖励下降等于鼓励砸地，直接破坏稳定性。

3. `signal_completeness`：best 版本已具备 forward progress（forward_vel × upright gate）和 stability（torso angle/ang_vel penalty）两个必要职责且可达；vertical 信号作为跳跃动力学的辅助引导并非缺失职责，但当前 `abs()` 形态与任务错位——它不区分主动推离和被动坠落，导致 proxy 与外部稳定性目标冲突。

4. `selected_level`：Level 2 —— `abs(vertical_vel)` 的数学形态直接导致 proxy 与外部任务错位（奖励坠落破坏稳定），证据明确否定该形态，不是单纯尺度问题。

5. `selected_intervention`：将 `abs(vertical_vel)` 替换为 `max(0, vertical_vel)`，仅奖励上升阶段（主动推离），不再奖励下降阶段（被动坠落）；系数从 0.2 降至 0.15 以匹配新值域（信号范围减半，保守起步）。

6. `falsifiable_hypothesis`：跳跃的主动控制集中在推离阶段，坠落由重力驱动不需额外激励；去除对下降的奖励后，agent 不应再被诱导砸地，终止率应大幅下降，episode length 应回到接近 1000 的截断水平。

7. `expected_next_round`：terminated 比例应从 19/20 显著下降到接近 0；episode_length 回升到 900+；score 应恢复到 2000+ 接近 best；`vertical_pushoff` 的 episode_sum_mean 应约为当前 `vertical_activity` 的 30-50%（系数降低 + 信号半波整流）。

8. `main_risk`：`max(0, vertical_vel)` 仍可能激励持续向上跳跃导致超出健康高度上限，若出现此问题下一轮需叠加高度门控或 bounded 归一化。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for hopping locomotion: forward progress + vertical push-off + stability.

    Components:
        forward_stability_reward – forward velocity weighted by torso uprightness
        vertical_pushoff         – reward only upward vertical velocity (push-off phase),
                                    gated by uprightness; does not reward downward motion
        stability_penalty        – penalty for torso tilt and angular velocity
    """
    # --- extract relevant signals from next_obs ---
    torso_angle = next_obs[1]          # rad, 0 = upright
    forward_vel = next_obs[5]          # positive = forward
    vertical_vel = next_obs[6]         # vertical velocity (positive = upward)
    torso_ang_vel = next_obs[7]        # rad/s

    # --- shared upright gate ---
    temp = 0.3
    upright_factor = 2.718281828 ** (-abs(torso_angle) / temp)

    # --- forward progress with upright gating ---
    forward_stability_reward = forward_vel * upright_factor

    # --- vertical push-off: reward only the active upward phase of hopping ---
    # max(0, vertical_vel) selects only upward motion (push-off),
    # ignoring the passive downward (landing) phase that gravity handles.
    # This prevents incentivising crash landings that destabilise the agent.
    w_pushoff = 0.15
    vertical_pushoff = w_pushoff * max(0.0, vertical_vel) * upright_factor

    # --- light stability penalty ---
    w_angle = 0.1
    w_ang_vel = 0.01
    stability_penalty = -w_angle * (torso_angle ** 2) - w_ang_vel * (torso_ang_vel ** 2)

    # --- total reward ---
    total_reward = forward_stability_reward + vertical_pushoff + stability_penalty

    components = {
        "forward_stability_reward": forward_stability_reward,
        "vertical_pushoff": vertical_pushoff,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```