# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取下一时刻的状态信息
    x = next_obs[0]   # 相对目标垫的水平位置
    y = next_obs[1]   # 相对目标垫的垂直位置（高度）
    vx = next_obs[2]  # 水平速度
    vy = next_obs[3]  # 垂直速度
    angle = next_obs[4]        # 机身倾角
    angular_vel = next_obs[5]  # 角速度

    # 主信号：鼓励接近目标垫
    distance = (x**2 + y**2)**0.5
    proximity_reward = -distance

    # 稳定约束：惩罚偏离竖立的倾角
    tilt_penalty = -0.5 * abs(angle)

    # 稳定约束：惩罚线速度大小，促使最终稳定
    speed = (vx**2 + vy**2)**0.5
    velocity_penalty = -0.1 * speed

    # 稳定约束：惩罚快速旋转
    rotation_penalty = -0.1 * abs(angular_vel)

    total_reward = proximity_reward + tilt_penalty + velocity_penalty + rotation_penalty

    components = {
        "proximity_reward": proximity_reward,
        "tilt_penalty": tilt_penalty,
        "velocity_penalty": velocity_penalty,
        "rotation_penalty": rotation_penalty
    }
    return total_reward, components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **proximity_reward**（主学习信号）  
   - 角色：驱动 agent 向目标着陆垫移动并停在其上。  
   - 形式：\(-\sqrt{x^2 + y^2}\)，每步都有连续梯度，距离越近奖励越大（越接近0）。  
   - 选取理由：任务核心是到达目标，距离负值是最直接、最稠密的引导信号。

2. **tilt_penalty**（稳定约束）  
   - 角色：鼓励机身保持竖直（角度→0），防止翻滚或异常姿态。  
   - 形式：\(-0.5 \cdot |\text{angle}|\)，权重较小，仅在偏离竖立时轻微惩罚。

3. **velocity_penalty**（稳定约束）  
   - 角色：促使 agent 在接近目标时减速，最终稳定停留。  
   - 形式：\(-0.1 \cdot \sqrt{v_x^2 + v_y^2}\)，轻量惩罚线速度，不会阻挡远处的高速接近。

4. **rotation_penalty**（稳定约束）  
   - 角色：抑制过快旋转，辅助姿态稳定。  
   - 形式：\(-0.1 \cdot |\omega|\)，权重与速度惩罚对称。

## 为何没有 terminal_success_reward / terminal_failure_penalty

- 环境卡片明确声明 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`，`info` 字典也为空，无法可靠获得任何成功/失败信号。因此本版本完全基于 step-level 的连续观测构建奖励，不加入终止事件相关的稀疏项。

## 留到后续迭代的组件

- **双腿接触相关的软着陆 proxy**：当前未使用 `left_support_contact` 与 `right_support_contact`，因为单独奖励接触容易被利用（如中途故意触地）。后续可通过组合距离、速度、角度与接触条件设计成多条件连续 proxy，作为任务完成信号的近似引导。
- **能量/动作代价**：v1 阶段不添加引擎使用惩罚，让 agent 先学会到达并稳定在目标垫，效率优化放在后续迭代。
- **距离门控的速度惩罚**：当前全局速度惩罚已经能促进稳定，但如果出现“远处不敢加速”的问题，可改为距离衰减的权重，让远处速度惩罚更轻。

## 训练后应观察的 failure mode

- **目标附近长时间震荡**：agent 可能围绕目标垫来回摆动而不完全停下。检查是否主信号梯度过于平缓或速度惩罚权重不足，需要微调权重或加入距离门控。
- **姿态翻滚**：如果 agent 在飞行过程中频繁大角度旋转，需要适当增大 `tilt_penalty` 或 `rotation_penalty` 的权重。
- **惰性不移动**：若所有惩罚使总奖励过于负向，agent 可能选择完全不动以避免惩罚。观察初始行为，若出现停滞，可考虑降低约束权重或提高主信号权重。