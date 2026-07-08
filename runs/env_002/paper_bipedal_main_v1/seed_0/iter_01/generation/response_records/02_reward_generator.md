# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重（可根据实际训练表现调整）
    w_forward = 1.0
    w_angle = 0.5
    w_vertical_vel = 0.1
    w_angular_vel = 0.05

    # 主学习信号：前进速度奖励
    forward_vel = next_obs[2]  # 水平速度，正值表示向前
    forward_reward = w_forward * forward_vel

    # 稳定/安全约束：抑制身体倾斜、跳动和高速旋转
    angle = abs(next_obs[0])           # 身体相对于竖直的角度
    vertical_vel = abs(next_obs[3])    # 上下跳动速度
    angular_vel = abs(next_obs[1])     # 身体转动角速度
    stability_penalty = - (w_angle * angle + w_vertical_vel * vertical_vel + w_angular_vel * angular_vel)

    total_reward = forward_reward + stability_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

- **forward_reward（主学习信号）**：直接奖励 `next_obs[2]` 的水平前进速度。  
  这是任务的核心驱动力：告诉智能体“向前走得越快，得分越高”。  
  速度信号每步都提供稠密梯度，与“走得尽可能远、尽可能快”的目标直接对齐。

- **stability_penalty（稳定/安全约束）**：对 `next_obs` 中的身体倾斜角度、垂直速度和角速度的绝对值进行惩罚。  
  它的角色是“方向盘”：防止智能体为了追求速度而大幅度侧倾或剧烈跳动，从而降低摔倒风险。  
  惩罚项的值明显小于前进奖励，避免让智能体不敢行动（over‑conservative policy）。

## 未使用 terminal_success_reward / terminal_failure_penalty 的原因

- 环境**没有提供显式的成功/失败标志**（`info` 为空，`explicit_success_flag_available = false`，`explicit_failure_flag_available = false`）。  
- `terminated` 信号无法在 `compute_reward` 中获取，因此无法判别是否到达终点或摔倒。  
- 摔倒的影响主要通过“摔倒后水平速度骤降”间接体现在 `forward_reward` 中，前一时刻的高惩罚（角度、速度异常）也起到早期纠正作用。

## 未在 v1 使用的骨架及其原因

- **alive_bonus**：需要 `done` 信息，但不可用；且可能诱导智能体原地站立获取小额积分。
- **energy_penalty / action_smoothness_penalty**：属于效率/平滑类约束，v1 阶段优先让智能体学会行走，后续迭代再考虑降低能耗和动作抖动。
- **contact_reward**：鼓励交替迈步的步态引导较复杂，且可能早期引入不必要的限制，推迟到后续迭代。

## 后续迭代预留

- 当基本行走稳定后，可加入：
  - 双腿交替接触奖励（`1 - |leg1_contact - leg2_contact|`）以塑造更自然的步态。
  - 轻量动作代价（`-λ * mean(action²)`）来降低能耗。
  - 若出现高频抖动，则引入动作平滑惩罚（需要存储上一步动作或近似差分）。

## 训练后应观察的 failure mode

- **fast_then_fail**：前向奖励权重过大，智能体学会猛冲后摔倒在地。  
  此时可适当增大 `w_angle` 或 `w_vertical_vel`。
- **hover_or_stand_still**：稳定性惩罚太重，智能体不敢迈步。  
  此时应减小 `w_angle` / `w_vertical_vel` / `w_angular_vel`，或提高 `w_forward`。
- **过度前倾定态**：智能体可能维持一个固定前倾角度并快速行走，但角度未在稳定性惩罚中产生足够压制。若行走稳健，此形态可接受；若经常因此摔倒，需加大角度惩罚。
