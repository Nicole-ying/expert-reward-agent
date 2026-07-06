# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v1 for hopping locomotion: encouraging forward progress while staying upright.
    Components:
        forward_stability_reward  – forward velocity weighted by torso uprightness
        stability_penalty         – penalty for torso tilt and angular velocity
    No terminal success/failure flags available, so no terminal rewards.
    """
    # --- extract relevant signals from next_obs ---
    torso_angle = next_obs[1]          # rad, 0 = upright
    forward_vel = next_obs[5]          # positive = forward
    torso_ang_vel = next_obs[7]        # rad/s

    # --- forward progress with upright gating ---
    # Exponential decay of the forward reward as the torso tilts.
    # Temperature controls how steeply the reward drops with tilt.
    temp = 0.3
    upright_factor = 2.718281828 ** (-abs(torso_angle) / temp)
    forward_stability_reward = forward_vel * upright_factor

    # --- light stability penalty ---
    # Penalise both the torso angle and its angular velocity to suppress
    # large oscillations while still allowing the hopping motion.
    w_angle = 0.1
    w_ang_vel = 0.01
    stability_penalty = -w_angle * (torso_angle ** 2) - w_ang_vel * (torso_ang_vel ** 2)

    # --- total reward ---
    total_reward = forward_stability_reward + stability_penalty

    components = {
        "forward_stability_reward": forward_stability_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及角色
1. **forward_stability_reward（主学习信号）**  
   - 角色：驱动agent持续向前跳跃，同时引导其保持躯干直立。  
   - 形式：`forward_velocity * exp(-|torso_angle| / temperature)`。  
   - 稠密性：每一步都提供梯度（forward_velocity连续，torso_angle连续）。  
   - 理由：直接奖励前进速度是locomotion的自然驱动力，但纯forward_velocity会导致快速但容易摔倒；将躯干角度作为衰减因子可引导policy在前进的同时维持直立，避免“fast‑then‑fail”。

2. **stability_penalty（稳定约束）**  
   - 角色：轻量抑制躯干大角度倾斜和快速旋转，减少失控摔倒。  
   - 形式：`-0.1 * torso_angle² - 0.01 * torso_angular_velocity²`。  
   - 权重较小，不会压制跳跃所需的姿态调整，但能抑制长时间大幅度摇摆。

## 为何不使用 terminal_success_reward / terminal_failure_penalty
- `explicit_success_flag_available = false`，`explicit_failure_flag_available = false`，且 `info` 字典为空。  
- 无法可靠判断回合是否因摔倒/超限终止，因此不能加入任何基于终止状态的奖励或惩罚。

## 其他未使用的骨架与理由
- **alive_bonus**：需要 `done` 标志判断是否存活，但环境中无此信号，无法实现。  
- **energy_penalty / action_smoothness_penalty**：v1阶段优先让agent学会跳跃前进，效率与动作平滑性留到后续迭代。  
- **stability_penalty (原骨架)** 的部分形式被采纳并变形为轻量约束。

## 留到后续迭代的组件
- 当策略能较稳定地向前跳跃后，可加入少量 `energy_penalty` 或 `torque_smoothness` 以提高能效和动作质量。  
- 若能通过 `next_obs` 推断地接触状态（例如 vertical_velocity 的符号变化），未来可设计 `contact_proxy` 辅助跳跃相位引导。

## 训练后应观察的 failure mode
- **站立不动或小幅度踏步**：若稳定性惩罚过强或forward奖励不足，agent可能选择微小动作避免摔倒，需调整 `upright_factor` 的温度或forward权重。  
- **快速移动但频繁摔倒**：提示稳定性约束不足或前进奖励未被充分抑制；可降低温度或增大角度惩罚。  
- **躯干持续大幅度摇摆**：需检查稳定性惩罚是否因权重太小而被forward奖励主导。  
- **利用垂直跳跃绕开前进奖励**：agent可能原地上下跳跃而不前进，但forward_velocity为零时奖励接近零，若无forward速度就不会获得主要奖励，故风险较低；一旦出现，可增加对零前进速度的隐性惩罚（但v1不必要）。
