# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 中提取关键信号（动作结果）
    hull_angle_next = next_obs[0]          # 躯干倾角，越小越稳
    hull_ang_vel_next = next_obs[1]        # 躯干角速度
    horizontal_velocity_next = next_obs[2] # 前向速度，正值向前

    # ---------- 健康门控：当躯干倾斜过大时自动衰减前向奖励 ----------
    danger_angle = 0.8   # 接近摔倒的阈值（~45°）
    max_angle = 1.2      # 完全关闭主奖励的阈值（~69°）
    # 线性衰减门：在 [0, danger_angle] 恒为 1，在 [danger_angle, max_angle] 从 1 线性降到 0
    gate = max(0.0, min(1.0, (max_angle - abs(hull_angle_next)) / (max_angle - danger_angle)))

    # ---------- 主学习信号：被门控的前向速度 ----------
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 稳定性约束：轻量角速度惩罚，抑制剧烈晃动 ----------
    w_ang_vel = 0.05
    stability_penalty = -w_ang_vel * (hull_ang_vel_next ** 2)

    # ---------- 总奖励与组件字典 ----------
    total_reward = velocity_gated + stability_penalty
    components = {
        "velocity_gated": velocity_gated,
        "stability_penalty": stability_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 选定的任务家族与动力学子类型
- **task_family**：locomotion_continuous_control（连续控制双足行走）
- **dynamics_subtype**：planar_biped，关节力矩驱动，在不平坦地形上前进，无固定目标点，体能健康（倾斜）决定存活。

## 选定的奖励职责（roles）
依据 environment_card.md 描述的任务目标“走得尽可能远、尽可能快，同时最小化能量消耗”，并结合可用信号与禁止标志，**v1 阶段选定以下职责**：

1. **主学习信号（mandatory）**：**前向推进** – 通过水平速度驱动向前运动。
2. **稳定/安全约束（conditional）**：**躯干倾斜控制** – 防止摔倒，采用健康门控形式而非独立惩罚。
3. **辅助稳定性约束（conditional）**：**躯干角速度抑制** – 减少剧烈晃动，提升步态质量。

## 职责–信号映射（role_to_signal_mapping）
| 职责 | 所用信号 | 位置 |
|------|----------|------|
| 前向推进 | 水平线速度 `horizontal_velocity` | `next_obs[2]` |
| 躯干倾斜控制 | 躯干倾角 `hull_angle` | `next_obs[0]` |
| 角速度抑制 | 躯干角速度 `hull_angular_velocity` | `next_obs[1]` |

所有信号均在 `next_obs` 中读取，反映动作执行后的状态，符合奖励可解释性。

## 公式算子选择与理由
- **主信号**：`dense_state_signal`（线性正奖励），形式 `w * signal`，但通过**门控因子 g** 变成 `gate * horizontal_velocity`。  
  - 选择线性是因为水平速度本身就是“前进距离/步”的天然梯度源，无需额外凸化。
  - 将门控融入主信号而非独立惩罚，实现了 `soft_health_gate` 算子：当躯干倾斜接近危险边界时，切断前进激励，迫使策略在健康区寻求速度，避免“先冲后摔”模式。
  - 门控采用**线性衰减** `max(0, min(1, (max_angle - |angle|) / margin))`，在安全区间内不干扰正常步态。
- **稳定性约束**：`quadratic_penalty`（二次惩罚）用于角速度。  
  - 权重极小（0.05），仅抑制突发性猛烈旋转，不影响正常姿态调节。  
  - 不采用 hinge，因为角速度小幅波动是步态自然的组成部分，不应只在“越界”时惩罚。

## 排除的职责及原因
- **terminal_success_reward**：环境无显式成功标志（`info` 为空，且 `explicit_success_flag_available=false`），不可用。
- **terminal_failure_penalty**：同样无显式失败标志，且 `done` 信号未按接口暴露，不能依赖。
- **动作能量/效率 (action_cost)**：v1 阶段优先学会稳定前进，能耗优化留到后续迭代。
- **joint_condition_proxy**（如“走完全程”）：缺少可组合的连续完成条件，且地形终点无信号，不适合在 v1 引入。
- **curriculum_weighting** 或 **dynamic gates**：训练初期无需阶段切换，v1 保持静态权重。
- **LIDAR 相关地形奖励**：环境核心目标是稳定行走，地形感知奖励可靠性低，会增加设计复杂度并干扰主要步态学习，故排除。

## 为何未使用 terminal_success_reward / terminal_failure_penalty
环境卡片明确指出：
- `explicit_success_flag_available: false`
- `explicit_failure_flag_available: false`
- `info` 为 `{}`，无法读取任何终止原因。

因此所有终止类奖励均无法在合法信号下实现。v1 完全依赖观测序列隐式推断风险（躯干倾角），用门控与软惩罚替代硬性失败信号。

## 留到后续迭代的职责
- 能量效率（关节力矩惩罚）  
- 步态质量（接触步序、双足同时离地惩罚）  
- 地形适应利用（LIDAR）  
- 关节极限软约束

## 训练后应观察的失败模式
- **velocity burst then fall**：agent 在短时间内高速冲撞后摔倒 → 说明门控衰减区间太窄或 max_angle 过高，需调整 danger_angle 或增加角速度惩罚权重。
- **stand still / gate exploitation**：agent 保持极小速度以维持完全竖直姿态，几乎不前进 → 门控过严或速度奖励权重不足，可能需要小幅抬高 danger_angle 或引入轻微适应项。
- **oscillatory gait**：高频率躯干摆动但未摔倒 → 角速度惩罚可能偏弱，可适当增大 `w_ang_vel` 或改为 hinge 形式。
- **侧向失衡未感知**：hull_angle 仅考虑竖直平面，若环境有侧向倾倒但观测未体现，该奖励将无法应对 → 需后续引入其他姿态信号（如果可用）。

总体而言，reward_v1 以门控速度为核心，搭配轻量角速度约束，在无任何终止标志的条件下，为双足步行提供了一条简洁、渐近的学习路径。
