# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主学习信号：前进速度（正向） ----
    forward_velocity = next_obs[13]  # body_x_velocity
    forward_reward = 2.0 * forward_velocity

    # ---- 稳定/健康约束：身体高度安全区间 ----
    body_height = next_obs[0]
    lower_safe = 0.3   # 终止边界 0.2 的 150%，留有缓冲
    upper_safe = 0.9   # 终止边界 1.0 的 90%
    height_penalty = (
        -5.0 * max(0.0, lower_safe - body_height) +
        -5.0 * max(0.0, body_height - upper_safe)
    )

    # ---- 稳定/健康约束：直立姿态 ----
    quat_x, quat_y = next_obs[2], next_obs[3]
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)  # 1 为完全直立
    upright_penalty = -1.0 * (1.0 - body_up_z)**2

    # ---- 辅助约束：侧向漂移抑制 ----
    lateral_velocity = next_obs[14]
    lateral_penalty = -0.5 * (lateral_velocity)**2

    # ---- 效率约束（极小权重）：动作能量代价 ----
    action_energy = sum(a**2 for a in action)
    energy_penalty = -0.01 * action_energy

    total_reward = forward_reward + height_penalty + upright_penalty + lateral_penalty + energy_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "height_health_penalty": height_penalty,
        "upright_orientation_penalty": upright_penalty,
        "lateral_drift_penalty": lateral_penalty,
        "action_energy_penalty": energy_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像对齐
- **task_family / dynamics_subtype**: `locomotion_continuous_control / multi_legged_body_locomotion`  
- **核心目标**: 最大化世界系 x 方向的前进速度，同时保持身体高度在安全区间和直立姿态以避免早停。

## 2. 已选角色与信号映射
根据环境卡片的 `reward_role_decomposition` 和 `role_to_signal_mapping`，选定以下角色：

| 角色 | 信号 | 公式算子 | 设计思路 |
|------|------|--------|--------|
| `forward_velocity_reward` (主学习信号) | `next_obs[13]` (body_x_velocity) | `dense_state_signal` 线性正奖励 | 直接驱动前进，权重 2.0 使正常步态下该分量占主导，提供每一步的稠密梯度 |
| `healthy_height_survival` (稳定约束) | `next_obs[0]` (body_z) | `dense_state_signal` 的 hinge 形式 | 仅在高度接近终止边界（<0.3 或 >0.9）时施加惩罚，权重 5.0 足够将机器人推回安全区间，但安全区内无干扰，避免过度限制步态 |
| `upright_orientation` (稳定约束) | `next_obs[2], next_obs[3]` 计算 `body_up_z` | `quadratic_penalty`（二次） | 鼓励身体保持直立，权重 1.0 对小幅倾斜（如步态自然摇晃）惩罚很轻，但在即将摔倒时才显著增大 |
| `lateral_drift_penalty` (辅助约束) | `next_obs[14]` (body_y_velocity) | `quadratic_penalty` | 抑制侧向漂移，权重 0.5 不会影响正常前进，但可防止策略发展出斜走或侧滑捷径 |
| `action_energy_penalty` (效率约束，极小权重) | `action` (8维扭矩) | `quadratic_penalty` | 动作维度 ≥6 时，按指导原则加入极低权重（0.01）的能耗项，避免完全无节制的扭矩输出，但不影响探索和力量型步伐的生成 |

## 3. 排除角色及原因
- **terminal_success_reward / terminal_failure_penalty**: 环境没有提供显式成功/失败标志（`explicit_success_flag_available=false`, `explicit_failure_flag_available=false`），且 `info` 已清空，无法获取终止原因。
- **distance_from_start / whole_trajectory_progress**: 绝对世界坐标被屏蔽，无可用信号。
- **contact_consistency_reward**: 足端接触力和触地标志不可用。
- **goal_reaching / sparse_event**: 任务非导航类型，没有目标位置。
- **vertical_oscillation_penalty**: v1 阶段优先保证前进和基础稳定，避免过度惩罚正常的步态起伏；后续迭代中若观察到明显跳跃行为可再加入。
- **action_smoothness_penalty**: 奖励函数无状态，无法计算动作变化率，无法实现差分平滑。

## 4. 公式选择理由
- 高度安全使用 **hinge penalty** 而非全域二次惩罚，因为高度存在明确的安全区间（0.2~1.0），使用 hinge 只在边界附近生效，避免在正常高度范围内（如 0.5~0.8）持续惩罚，防止 agent 滞留在某个“完美高度”而放弃前进。
- 直立姿态和侧向漂移使用 **quadratic penalty** 而非门控：它们在正常步态下误差已经很小，全域二次施加轻微的梯度可加速收敛，不会压制探索。
- 前进速度保持 **线性** 形式：预期速度范围在合理限值内，不会出现无界膨胀，且凸化会改变梯度语义，留待后续调优。

## 5. 设计原则自查
- **稠密性**: 每个组件每一步都提供有意义的梯度，无稀疏信号。
- **尺度平衡**: 在合理的前进速度（~2 m/s）下，主奖励约 4.0，各惩罚项合计约 -0.3，总奖励保持在正且由主目标驱动，约束项仅起细调作用。
- **信号冲突**: 没有使用两个计算同一物理量的信号，也未使用 `original_reward` 或 `info`。
- **可利用风险**: 
  - 只奖励速度可能引发 agent 前冲后摔倒，但 hinge 高度惩罚和直立惩罚提供了“健康刹车”，迫使 agent 在接近危险前放缓。
  - 只使用生存相关奖励不会诱导静止，因为主目标直接关联速度，且无存活分激励原地不动。

## 6. 后续迭代方向
- 观察 **failure modes**: 若策略频繁在高速下摔倒（先冲后死），可将高度/姿态惩罚升级为 `soft_health_gate`，在健康恶化时衰减前进奖励。
- 若前进速度卡在较低平台，可对前进分量进行凸化（如 `forward_velocity**2`）。
- 出现明显上下颠簸时，适度加入 `vertical_oscillation_penalty`。
- 训练后期可逐步增加 `action_energy_penalty` 或引入 `action_smoothness`（需要状态记忆），以优化能耗和动作流畅度。