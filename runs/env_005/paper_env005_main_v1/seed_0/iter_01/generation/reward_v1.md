# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from the result state (next_obs)
    body_z = next_obs[0]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]

    # Weights
    w_fwd = 1.0
    w_lat = 0.5
    w_ang = 0.001
    w_h = 10.0

    # 1. Main learning signal: forward velocity reward
    forward_reward = w_fwd * body_x_vel

    # 2. Stability constraint (penalty for dangerous/inefficient deviations)
    lateral_penalty = w_lat * abs(body_y_vel)
    angular_penalty = w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)
    height_penalty = w_h * (
        max(0.0, 0.25 - body_z) + max(0.0, body_z - 0.95)
    )
    stability_penalty = -(lateral_penalty + angular_penalty + height_penalty)

    total_reward = forward_reward + stability_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及角色

- **forward_velocity_reward** （主学习信号）  
  直接奖励机器人沿世界 x 轴（前进方向）的速度 `body_x_vel`。  
  这是一个稠密、每步都有梯度的信号，驱使机器人持续向前移动。权重设为 1.0，使其成为策略学习的主要驱动力。

- **stability_penalty** （稳定/安全约束，组合容器）  
  将三项与安全性与运动质量相关的惩罚打包成一个组件，避免堆叠过多零散项。内部包含：
  - **横向速度惩罚**：抑制侧向漂移 (`|body_y_vel|`)，权重 0.5，防止机器人“横着走”。
  - **角速度惩罚**：惩罚过快的滚转、俯仰和偏航角速度（平方和），权重 0.001，促使姿态平稳，避免强烈旋转导致失控。
  - **高度软边界惩罚**：当身体高度 `body_z` 低于 0.25 或高于 0.95 时施加线性惩罚，权重 10.0。这个区间比终止界限（0.2~1.0）更严格，提供预警梯度，防止因高度越界提前死亡，同时不干扰正常行走时的自然高度变化。

总奖励 `total_reward = forward_reward + stability_penalty`，两个组件角色分明。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

环境卡片明确声明 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`，且 `info` 字典为空，不存在任何终止原因或成功/失败标志。因此任何依赖显式 flag 的稀疏奖励均不可用于 v1。

## 为什么未使用其他骨架建议的组件

- **alive_bonus**：没有显式 done flag（环境只通过截断或终止条件完成，但我们的函数没有 done 输入），且每步存活奖励容易导致“站立不动”的策略，与前进目标冲突。
- **energy_penalty**（动作扭矩惩罚）与 **action_smoothness_penalty**：v1 阶段先让机器人学会稳定前进，后续再优化效率与平滑性；且缺乏动作历史信息，无法实施平滑惩罚。
- **姿态对齐奖励**（利用四元数计算直立投影）：虽然直观，但四元数存储顺序（w,x,y,z 或 x,y,z,w）未在卡片中明确，为避免错误采用更稳健的角速度惩罚替代；姿态约束的精细化准备在后续迭代中加入。
- **足部接触奖励**：离散接触信号容易被利用（如原地踏步），且需要与前进信号精细配合；v1 暂不引入，后续可考虑作为步态引导的辅助项。

## 后续迭代可加入的组件

- 身体直立姿态奖励（基于 quaternion 投影，待确认四元数格式后实现）。
- 动作能量或平滑度惩罚，用于提升步态效率。
- 动态速度目标或课程：逐步提高期望前进速度，克服“慢速安全”策略。
- 足部接触模式约束，辅助生成对称、周期性的步态。

## 训练后应观察的 failure mode

1. **standing still（原地站立不动）**  
   如果 `stability_penalty` 的权重过强，agent 可能学会保持高度安全、姿态稳定的静止姿态，`forward_reward ≈ 0`，但总体奖励仍然为正（`stability_penalty ≈ 0`）。应对策：在可视化中监控前进速度的均值，若过低则适当提高 `w_fwd` 或逐步降低高度/角速度惩罚。

2. **slow cautious walking（过度保守的慢行）**  
   策略以小速度前进，惩罚项几乎为 0，获得微小的净奖励。这比摔倒强，但性能受限。随后迭代可引入增加期望速度目标或能量惩罚来推动更快行走。

3. **fast‑then‑fall（快速前进后失稳）**  
   过度追逐前进奖励导致姿态剧烈波动或身体高度越界，episode 提前终止；此时平均回报不高。应在训练曲线中观察 episode length 是否频繁缩短。若发生，可增大角速度惩罚或高度惩罚权重，或加入姿态对齐奖励以增强稳定性。

4. **lateral drift（侧向漂移）**  
   机器人可能边前进边侧移，虽完成前向移动但路径弯曲。横向速度惩罚已存在，如果漂移仍明显，可适当增大 `w_lat`。