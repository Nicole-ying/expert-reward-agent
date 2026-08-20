# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component 1: forward speed tracking (main learning signal)
    # Encourage body_x_velocity to stay close to a target speed.
    # ------------------------------------------------------------
    target_speed = 1.5  # m/s, modest running pace
    speed_error = abs(next_obs[13] - target_speed)
    forward_tracking = -10.0 * speed_error

    # ------------------------------------------------------------
    # Component 2: stability cost bundle
    # Penalise deviations that lead to falls or inefficient gaits.
    # ------------------------------------------------------------
    # height: target 0.6 (middle of healthy range 0.2-1.0)
    height_error = next_obs[0] - 0.6
    height_cost = 5.0 * (height_error ** 2)

    # orientation: penalise tilt using the first two quaternion
    # components (x, y); they are near zero when the robot is upright.
    tilt_cost = 2.0 * (next_obs[2] ** 2 + next_obs[3] ** 2)

    # lateral velocity: suppress side‑ways motion
    lateral_cost = 1.0 * abs(next_obs[14])

    # angular velocity (roll + pitch only)
    angular_cost = 0.1 * (abs(next_obs[16]) + abs(next_obs[17]))

    stability_cost = height_cost + tilt_cost + lateral_cost + angular_cost

    # ------------------------------------------------------------
    # Total reward
    # ------------------------------------------------------------
    total_reward = forward_tracking - stability_cost

    components = {
        "forward_tracking": forward_tracking,
        "stability_cost": -stability_cost
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 核心理念
前几次尝试几乎都使用了“最大化前进速度”作为主信号，但效果不理想——机器人要么因过度追求高速而摔倒，要么在倒地的边缘获得正向信号。本设计改用**速度跟踪**代替无上限的速度奖励：设定一个适中的目标速度（1.5 m/s），机器人越接近目标速度得分越高，超过或不足都会受到惩罚。这样既能推动机器人向前运动，又避免了对极端速度的盲目追求，使策略更容易在稳定状态下收敛。

## 组件角色
1. **forward_tracking（主学习信号）**  
   - 角色：核心前进驱动力，约束速度大小并保持匀速。  
   - 数学：`-10.0 * | body_x_velocity - 1.5 |`。  
   - 稠密性：每一步都有连续梯度，且饱和非常平滑。  
2. **stability_cost（稳定约束组合）**  
   - 角色：将多个低维安全/姿态约束打包为一个轻量惩罚项，防止崩溃。  
   - 子项：  
     - 高度偏差（目标 0.6 m，二次）  
     - 身体倾斜（基于四元数前两分量平方和，直立时≈0）  
     - 侧向速度绝对值  
     - 滚转/俯仰角速度绝对值  
   - 权重均较小，起到“方向盘”作用，不会压倒主信号。

## 未使用 terminal_success_reward / terminal_failure_penalty 的原因
环境没有提供显式的 success 或 failure 标志（explicit_*_flag_available = false），info 字典为空，因此无法可靠地发放终点奖励或惩罚。强行使用会引入噪声，甚至鼓励错误的终止策略。

## 留到后续迭代的组件
- **能量/动作代价**：v1 阶段先让机器人学会稳定行走，不引入动作幅度惩罚，以免“不敢动”。  
- **脚步接触奖励**：步态模式的连续鼓励（例如交替支撑）仍是一个有前途的方向，但需要更复杂的多条件组合，更适合后续引入。  
- **活命奖励**：缺少 done 标志，常数值无学习意义，暂时略过。

## 训练后需要观察的失败模式
1. **静止不动或微弱前进**：速度跟踪惩罚驱动，理论上会推动机器人达到目标速度，但如果稳定性惩罚过强，机器人可能选择低幅度、低风险的慢速移动。此时应微调 `forward_tracking` 的权重或适当放宽目标速度。  
2. **高度震荡**：机器人可能在 0.6 m 附近上下振荡以平衡高度成本与速度跟踪，需观察高度轨迹是否平滑。  
3. **转向惩罚不足**：当前未惩罚偏航角速度，机器人可能逐渐转向，但环境无绝对朝向奖励，可接受。  
4. **目标速度不当**：如 1.5 m/s 对机器人太困难或太容易，后续可通过实验调整。
