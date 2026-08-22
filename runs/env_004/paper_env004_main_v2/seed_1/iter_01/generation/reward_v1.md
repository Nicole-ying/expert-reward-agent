# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V1 reward for single-leg hopper continuous locomotion.
    Drives forward progress while keeping the torso upright and rotationally stable.
    """
    # Weights
    w_fwd = 1.0          # main progress drive
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty

    # Main learning signal: forward velocity (after transition)
    # Clip to non‑negative so backward exploration is not penalised.
    forward_progress = w_fwd * max(0.0, next_obs[5])

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    total_reward = forward_progress + torso_angle_penalty + torso_angular_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 奖励组件与角色

本奖励函数包含 **三个组件**，符合 V1 阶段 2~4 个组件的预算约束：

1. **forward_progress（主学习信号）**  
   - 角色：驱动agent持续向前跳跃移动。  
   - 形态：`w_fwd · max(0, next_obs[5])`，使用转移后的水平速度，并截断负值，避免惩罚偶然的后退探索，鼓励所有能产生正向位移的策略。  
   - 每步都有梯度（在线性区），与任务目标直接相关。

2. **torso_angle_penalty（稳定约束）**  
   - 角色：保持躯干直立，防止过度倾斜导致过早终止。  
   - 形态：`-w_angle · |next_obs[1]|`，惩罚躯干倾角的绝对值，将agent约束在直立状态。  
   - 提供持续梯度，防止快速前倾/后仰造成的失控。

3. **torso_angular_velocity_penalty（稳定约束）**  
   - 角色：抑制躯干剧烈旋转，增加姿态平滑度。  
   - 形态：`-w_angvel · |next_obs[7]|`，惩罚角速度的幅度。  
   - 权重远小于主信号，仅起“方向盘”作用，避免因旋转太快而轻易摔倒。

## 未使用 terminal_success_reward / terminal_failure_penalty 的原因

- 环境 **没有显式成功或失败标志**（`explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 `false`）。  
- `info` 字段完全不可用，无法可靠区分失败、超时或正常截断。  
- 因此任何基于终止标志的奖励或惩罚都违反可用性原则，V1 完全不引入。

## 后续迭代可加入的组件

- **energy_penalty / action_smoothness_penalty**：待agent学会稳定前向跳跃后，再约束力矩幅值或动作抖动，提升能效。  
- **height_deviation_penalty**：当环境提供合理的目标高度或健康区间时，可加入身体高度约束，进一步引导正常跳跃姿态。  
- **survival-based shaping**：如果后续能获得可靠的存活步数信息，可将存活激励作为辅助锚点，但需谨防agent“原地站立”的漏洞。

## 训练后预期观察的 failure mode

1. **fast_then_fail** – agent可能通过极大向前冲量获得高进度分，但因姿态失控而快速结束。需调整权重或引入速度上限。  
2. **hover_or_stand_still** – 若角度/角速度惩罚过强，agent可能选择几乎不动以避开惩罚，此时应降低稳定惩罚或提高主信号权重。  
3. **over_conservative_policy** – 同样因稳定约束过强，agent可能采用极小步幅的蹑手蹑脚模式，前进极慢。可稍减惩罚或加入最小速度要求。  
4. **non‑hopping cheat** – 在模拟器中，agent可能找到一种保持低重心滑行或拖拽脚部的方式获得前向速度而不真正跳跃。需要后续通过高度或接触信息（若可用）来识别并纠正。