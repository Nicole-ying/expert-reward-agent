# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Main learning signal: forward velocity
    forward_vel = next_obs[2]                 # positive = moving forward
    forward_progress = 1.0 * forward_vel

    # Stability: keep torso upright
    hull_angle = next_obs[0]                 # radians, 0 = vertical
    tilt_penalty = -5.0 * (hull_angle ** 2)

    # Stability: suppress rotational jitter
    hull_angvel = next_obs[1]               # rad/s
    angular_velocity_penalty = -1.0 * (hull_angvel ** 2)

    # Efficiency: light action cost
    avg_action_power = (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2) / 4.0
    action_cost = -0.05 * avg_action_power

    total_reward = forward_progress + tilt_penalty + angular_velocity_penalty + action_cost

    components = {
        'forward_progress': forward_progress,
        'tilt_penalty': tilt_penalty,
        'angular_velocity_penalty': angular_velocity_penalty,
        'action_cost': action_cost
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## Selected task family / dynamics subtype
- **task_family**: `locomotion_continuous_control`
- **dynamics_subtype**: `planar_bipedal_gait`（2D双足步行，连续力矩控制）

## Selected reward roles
| Role | Component name | Weight | Signal source | Formula operator |
|------|----------------|--------|---------------|------------------|
| 主学习信号 | `forward_progress` | 1.0 | `next_obs[2]` (horizontal velocity) | `dense_state_signal` (positive linear) |
| 稳定约束 | `tilt_penalty` | -5.0 | `next_obs[0]` (hull angle) | `quadratic_penalty` |
| 稳定约束 | `angular_velocity_penalty` | -1.0 | `next_obs[1]` (hull angular velocity) | `quadratic_penalty` |
| 效率/动作代价 | `action_cost` | -0.05 | `action[0:4]` (joint torques) | `quadratic_penalty` (mean of squares) |

## Role-to-signal mapping
- `forward_progress` → `next_obs[2]`（前进方向水平线速度，可直接驱动“更快更远”）
- `tilt_penalty` → `next_obs[0]`（躯干与垂线夹角，越小越稳）
- `angular_velocity_penalty` → `next_obs[1]`（躯干角速度，抑制旋转抖动）
- `action_cost` → 四个关节力矩平方均值（降低能耗）

## Excluded roles and reasons
- **terminal_success_reward / terminal_failure_penalty**  
  `explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 `false`，`info` 为空字典，无法获得任何终止信息。因此不能设计基于终端事件的奖励/惩罚。
- **vertical_velocity_constraint**  
  虽然 `next_obs[3]` 可用于抑制跳跃/重落地，但受限于 v1 的 **role-based component budget**（推荐 2~4 个组件，其中稳定约束最多 2 个）。当前已选择倾斜与角速度作为最直接的“防摔倒”约束，垂直速度约束留到后续迭代。
- **contact coordination / gait symmetry**  
  `next_obs[8]` 和 `next_obs[13]` 提供腿接触信息，但步态协调属于中级技能，v1 优先学会“稳定前进”而非精细步态。预留到 v2 或之后。
- **task completion proxy**  
  环境没有位置/到终点距离信息，无法构造合理的“接近终点”连续信号；强行使用二值乘积 proxy 会退化为稀疏奖励，不符合 v1 稠密性原则。

## Design rationale summary
- 主信号采用 **正向速度线性奖励**，每步提供稠密梯度，引导智能体向前移动。
- 躯干倾斜和角速度的 **二次惩罚** 直接映射终止条件（摔倒）的前兆：倾斜角过大或剧烈旋转会导致摔倒。二次形式在安全范围内惩罚很小，越接近危险区域惩罚急剧增大，起到 soft constraint 作用。
- **动作代价** 权重很小（0.05），避免压制探索，同时为后续效率优化留出接口。
- 所有组件均基于 `next_obs` 和 `action`，不依赖 `info`、`original_reward` 或终止标志，符合环境接口契约。
- 各组件量级已做初步平衡：前进速度 1.0 m/s 贡献 1.0，倾斜角 0.2 rad 惩罚约 0.2，角速度 1.0 rad/s 惩罚 1.0，动作平均功率约 0.5 惩罚 0.025。前进仍然是主导驱动力，稳定约束只在姿态恶化时显著干预。

## What is left for future iterations
- **vertical velocity hinge**：当跳跃或硬着陆成为明显问题且 agent 学会基本行走后，可以加入 `hinge(｜vz｜ > 0.3)` 约束。
- **soft health gate**：若出现“先冲后摔”模式，可将倾斜角映射为门控因子乘到 `forward_progress` 上，在危险姿态时衰减前进奖励。
- **contact coordination**：利用双足触地信号鼓励交替迈步，改善步态自然度。
- **curriculum / dynamic weighting**：如有训练进度信号，可后期平滑切换权重。

## Expected failure modes to watch during training
1. **slow‑creep policy**：智能体为了减少倾斜/角速度惩罚而故意慢走甚至原地踏步，前进速度始终很低。
2. **forward‑fall exploit**：为了追求高速度奖励，智能体大幅前倾但仍能保持几秒不摔，获得高额累计奖励，这种策略在摔倒后没有额外惩罚（因为无法在奖励里访问终止）可能导致它不断尝试危险前倾。
3. **excessive joint torque**：动作代价较轻，早期可能出现高频大幅抖动的控制策略。
4. **punishment dominance**：如果稳定约束公式过于严厉（权重过高），智能体可能完全放弃前进，仅维持直立。可通过调整权重或改用 hinge 解决。