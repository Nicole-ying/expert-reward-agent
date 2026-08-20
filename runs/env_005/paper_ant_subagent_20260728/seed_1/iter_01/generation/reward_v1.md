# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_w = obs[1]
    quat_x = obs[2]
    quat_y = obs[3]
    quat_z = obs[4]
    body_x_velocity = obs[13]
    # body_y_velocity = obs[14]  # reserved for later
    # body_z_velocity = obs[15]  # reserved for later

    # ---------- Compute derived signals ----------
    # Uprightness proxy: component of body z-axis in world z (1.0 = perfectly upright)
    # From quaternion: body_up_z = 1 - 2*(quat_x**2 + quat_y**2)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Reward components ----------
    # 1) Forward progress (main learning signal)
    # Only reward positive forward velocity; zero or backward yields 0
    forward_progress = max(0.0, float(body_x_velocity))
    reward_forward = 1.0 * forward_progress

    # 2) Body height penalty (hinge at boundaries)
    # Healthy range is (0.2, 1.0). We add penalty when approaching the edges:
    # lower danger: z < 0.25, upper danger: z > 0.95
    height_low = max(0.0, 0.25 - body_z)
    height_high = max(0.0, body_z - 0.95)
    reward_height_penalty = -5.0 * (height_low + height_high)

    # 3) Upright stability penalty
    # Penalise deviations from perfect uprightness (body_up_z = 1.0)
    upright_error = 1.0 - body_up_z  # non-negative when body_up_z <= 1.0
    reward_upright = -2.0 * upright_error

    # ---------- Total reward ----------
    total_reward = reward_forward + reward_height_penalty + reward_upright

    components = {
        "forward": reward_forward,
        "height_penalty": reward_height_penalty,
        "upright_penalty": reward_upright
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 任务族与动态子类型
- `task_family`: locomotion_continuous_control
- `dynamics_subtype`: multi_legged_body_locomotion (四足扭矩控制，无显式接触力)

## 选定的奖励职责
根据 `reward_role_decomposition` 中的强制性职责，选择了三个核心职责：

1. **forward_progress** – 主学习信号  
   - **信号**: `body_x_velocity` (obs[13])  
   - **公式算子**: `dense_state_signal` 的 positive 形式，但使用 `max(0, velocity)` 以避免奖励后退，避免 agent 误以为后退也能得分。  
   - **权重**: 1.0，作为任务驱动力的主体。

2. **body_height_penalty** – 高度边界软约束  
   - **信号**: `body_z` (obs[0])  
   - **公式算子**: `dense_state_signal` 的 hinge 形式，在 0.25 ~ 0.95 的外部施加惩罚，既提供安全区内的无罚空间，又提前警告即将越界。这与二次惩罚不同，不会在健康中心区域施加不必要的阻尼。  
   - **权重**: 5.0，使边界惩罚能够与前进奖励的量级大致平衡。

3. **upright_stability** – 直立维持约束  
   - **信号**: 由四元数计算的 `body_up_z = 1 - 2*(quat_x² + quat_y²)`  
   - **公式算子**: 线性惩罚 `-w * (1 - body_up_z)`，提供持续但轻量的梯度，防止严重倾斜。  
   - **权重**: 2.0，使微小倾斜（≈0.1 误差）产生的惩罚不会压制前进信号，但较大倾斜会被明显惩罚。

## 未使用或排除的职责
- **横向速度惩罚 (lateral velocity)** 与 **垂直速度惩罚 (vertical velocity)**：属于次要目标，留到后续迭代（v2+）再引入，v1 集中学习向前移动和基础稳定。  
- **动作/能量效率**：v1 阶段不引入，避免过早抑制动作探索，保持简单。  
- **terminal_success_reward**：环境中不存在显式成功标志 (`explicit_success_flag_available=false`)，因此不设计终态成功奖励。  
- **terminal_failure_penalty**：同样理由，无显式失败标志，且高度越界通过每步 hinge 惩罚已提供连续警告，无需额外稀疏终态惩罚。  
- **soft_health_gate / joint_condition_proxy / curriculum_weighting**：v1 不采用复杂门控与动态权重，待观察到具体训练问题后（如高 terminated 率下主奖励仍很大）再考虑引入。

## 观测与信息使用说明
- 只使用 `obs` 中的允许字段，未使用 `next_obs`（当前设计已够）。  
- 完全未触碰 `original_reward` 和 `info`，符合接口契约。

## 设计自检
- 所有终止条件（高度越界、姿态发散）都有对应的前兆软信号（hinge、upright 惩罚）。  
- 任务目标直接映射到每步梯度（前进速度）。  
- 动作维度为 8，暂时未添加效率项，留到后续。  
- 各组件量级：前进速度典型值约 0.5~2.0，奖励 0.5~2.0；高度惩罚在偏离 0.05 时产生 0.25；直立误差约 0.0~0.2 时惩罚 0.0~0.4；总体尺度可比，不会出现某个组件统治。

## 后续迭代关注点
- 若机器人倾向于低速小步走（reward 停滞），可将前进速度奖励改为凸化形式（如 `max(0, velocity)**2`）。  
- 若因过度追求速度导致频繁摔倒，可引入 soft_health_gate 将高度/姿态状态乘到前进奖励上，而非仅依赖惩罚。  
- 引入横向/垂直速度轻约束，以及动作平滑惩罚，提升步态质量和能效。