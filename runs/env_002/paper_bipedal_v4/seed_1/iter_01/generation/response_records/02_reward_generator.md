# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === forward progress (main learning signal) ===
    forward_vel = next_obs[2]  # horizontal_velocity
    forward_reward = 1.0 * max(0.0, forward_vel)

    # === energy penalty (secondary objective: minimize joint torques) ===
    torque_sum_sq = (
        action[0] ** 2 + action[1] ** 2 +
        action[2] ** 2 + action[3] ** 2
    )
    energy_penalty = -0.01 * torque_sum_sq

    # === upright stability (avoid falling) ===
    hull_angle = next_obs[0]
    hull_angvel = next_obs[1]
    upright_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angvel ** 2)

    total_reward = forward_reward + energy_penalty + upright_penalty
    components = {
        "forward": forward_reward,
        "energy": energy_penalty,
        "upright": upright_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务类型
- **task_family:** `locomotion_continuous_control`  
- **dynamics_subtype:** `planar_bipedal_gait`

## 选用的奖励职责（roles）

| 职责 | 角色 | 是否必须 | 使用信号 | 公式算子 | 理由 |
|------|------|---------|----------|----------|------|
| `forward_progress` | 主学习信号 | 必须 | `next_obs[2]`（水平速度） | `dense_state_signal`（正速度部分，`max(0, v)`） | 驱动智能体持续向前行走，直接对应任务目标 |
| `energy_penalty` | 效率/动作代价 | 必须 | `action[0..3]`（关节力矩） | `quadratic_penalty`（力矩平方和惩罚） | 环境需求明确要求最小化能耗，权重极低，避免压制探索 |
| `upright_stability` | 稳定/安全约束 | 必须 | `next_obs[0]`（躯干倾角）、`next_obs[1]`（躯干角速度） | `quadratic_penalty`（角度和角速度的平方惩罚） | 防止摔倒终止训练，提供轻量姿态约束 |

## 信号映射与公式选择

- **forward_progress:** 直接奖励正的水平速度，负速度无奖励但也不惩罚，防止奖励倒退。使用 `max(0.0, next_obs[2])` 实现稠密且有意义的梯度。
- **energy_penalty:** 对四个关节力矩平方求和，乘以系数 `-0.01`。该尺度远小于前向奖励（通常 0~0.02），确保任务驱动力不被能耗项压制。
- **upright_stability:** 分离的角度惩罚 `-0.5 * angle²` 和角速度惩罚 `-0.1 * angvel²`。正常行走时，角度/角速度均较小，惩罚在 `0~0.005` 量级，不影响前向奖励；一旦倾斜变大，惩罚迅速上升，能及时促使恢复。

## 排除的职责及原因

- **`contact_regularization`（条件职责）**：需要定义期望接触模式，早期可能引入“原地踏步”捷径；留到后续观察到步态异常再启用。
- **`terrain_aware_adaptation`**：lidar信号解释性强，初期禁用。
- **`absolute_distance_reward`**：环境未提供x坐标，无法可靠计算。
- **`alive_bonus`**：鼓励静止，与“快且远”目标冲突。
- **`goal_reaching_bonus`**：无显式成功标志，info为空，不可靠。
- **`terminal_success_reward` / `terminal_failure_penalty`**：`explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，强行使用会制造虚假信号。

## 后续迭代考虑

- 如果出现**原地踏步**或**单腿拖拽**，可加入 `contact_regularization` 以鼓励交替接触。
- 如果复杂地形上适应不足，可引入地形感知辅助项，但需谨慎处理lidar噪声。
- 若动作剧烈颤动，可加入动作变化率惩罚（需保存上一步动作）。
- 能耗项权重可随训练进度微调，但v1先用固定轻量惩罚。

## 预期观察的典型失败模式

1. **快速摔倒**：角度/角速度惩罚若未能及时起效，需上调 `upright_penalty` 权重。
2. **完全不进（零/负速度）**：检查 `energy_penalty` 是否过强，可暂降能耗系数或将前向权重翻倍。
3. **原地踏步但稳定**：增大 `forward_progress` 系数，后期考虑加入交替接触奖励。
4. **颤动高能耗低速度**：可引入动作平滑代价（v2），或降低力矩惩罚系数。
5. **仅适用于平坦地形**：未来加地形感知小奖励或curriculum。
