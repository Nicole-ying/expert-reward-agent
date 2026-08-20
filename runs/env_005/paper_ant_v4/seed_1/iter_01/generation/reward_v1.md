# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract signals ----------
    body_z          = obs[0]
    qw, qx, qy, qz = obs[1], obs[2], obs[3], obs[4]
    body_x_vel      = obs[13]
    body_y_vel      = obs[14]
    roll_vel        = obs[16]
    pitch_vel       = obs[17]

    # compute torso uprightness (world z projection)
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)

    # -------- 1. forward progress (bounded) --------
    forward_reward = 2.0 * (body_x_vel / (1.0 + abs(body_x_vel)))

    # -------- 2. health height (soft boundary cost) --------
    height_penalty = 0.0
    if body_z < 0.3:
        height_penalty = -20.0 * (0.3 - body_z)
    elif body_z > 0.9:
        height_penalty = -20.0 * (body_z - 0.9)

    # -------- 3. upright stability (posture + angular velocity) --------
    upright_reward = 1.0 * body_up_z
    ang_penalty    = -0.005 * (roll_vel**2 + pitch_vel**2)
    upright_stability = upright_reward + ang_penalty

    # -------- 4. lateral drift penalty --------
    lateral_penalty = -0.1 * (body_y_vel**2)

    # -------- total reward and components --------
    total = forward_reward + height_penalty + upright_stability + lateral_penalty

    components = {
        "forward_progress":       forward_reward,
        "height_penalty":         height_penalty,
        "upright_stability":      upright_stability,
        "lateral_drift_penalty":  lateral_penalty
    }

    return float(total), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- `locomotion_continuous_control` / `multi_legged_body_locomotion`  
  任务核心是四足机器人持续向前运动，同时保持身体高度与姿态直立。

## selected reward roles
根据 `reward_role_decomposition` 的 mandatory 和 conditional 职责，v1 选取以下角色：

| role_id | 说明 | 信号 | 公式算子 |
|---|---|---|---|
| forward_progress | 主学习信号，激励持续正向平移 | `body_x_velocity` (obs[13]) | **bounded_signal**：`wx / (1 + |x|)` 正向渐近奖励，负向惩罚，防止无界速度值 |
| health_height | 维持身体高度在安全区间 (0.2, 1.0) 内，避免提前终止 | `body_z` (obs[0]) | **linear margin penalty**：在舒适区 [0.3, 0.9] 外施加与违反量成正比的惩罚，提供梯度又不过于严厉 |
| upright_stability | 保持躯干直立，抑制大幅摇晃和翻滚 | `body_up_z` (由四元数计算)、`roll_vel`, `pitch_vel` | **直接姿态奖励** + **二次角速度惩罚**，将姿态鼓励与稳定性约束合并为一个组件 |
| lateral_drift_penalty | 轻微惩罚侧向漂移，使前进更直线 | `body_y_velocity` (obs[14]) | **二次惩罚**，权重极小，避免阻碍早期探索 |

## role_to_signal_mapping
- `forward_progress` ← `body_x_velocity`
- `health_height` ← `body_z`
- `upright_stability` ← `quat {x,y}` → `body_up_z`， `roll_vel`, `pitch_vel`
- `lateral_drift_penalty` ← `body_y_velocity`

所有信号均在 `obs` 中直接或通过组合可得，`info` 完全不可用，未依赖任何未声明字段。

## excluded roles 及原因
- **survival_bonus**：缺少显式存活 flag（info 屏蔽），无法可靠实现。
- **goal_distance_progress**：绝对 x/y 位置不可用，不能计算距离进展。
- **contact_force_penalty**：环境未提供接触力传感器。
- **posture_mimicry**：无参考关节角度/轨迹，且自主发现步态更鲁棒。
- **action_magnitude_regularization**：条件职责，v1 排除，避免早期抑制扭矩探索；留到后续训练步骤引入。
- **terminal_success_reward / terminal_failure_penalty**：不存在显式 `success`/`failure` flag，`info` 为空，无法可靠辨识，且易误导。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境卡片明确指出 `explicit_success_flag_available: false` 且 `info` 全部禁止，无法获得 episode 级的成功或失败信号。因此 v1 完全通过每步连续状态信号驱动学习和避免过早终止。

## 哪些职责留到后续迭代
- **动作平滑/正则化（action_magnitude / smoothness）**：待前进步态基本稳定后逐步加入，以提升能效和动作连贯性。
- **动态课程/门控（curriculum / gated reward）**：现阶段无需，且 `training_progress` 不宜过早依赖。
- **更复杂的多条件完成近似 proxy**：当前仅通过直立和高度保护健康，尚未引入“接近目标”的软完成信号。

## 训练后应重点观察的 failure modes
- **无前进运动（stand‑still）**：虽然高度和姿态可能有分，但 `forward_progress` 应驱动前进；若 agent 收敛到静止说明权重失衡或探索不足。
- **跌倒/翻滚（falling）**：当高度惩罚和直立奖励不足时，agent 可能选择向前翻滚获得前进分数，需观察 `body_z` 和 `body_up_z` 下降趋势。
- **剧烈抖动（jitter）**：角速度惩罚权重若不够，可能导致高频振荡但近似直立的策略。
- **撞高度上限（launching）**：过强的前进驱动配合不合适步态可能使机器人弹离地面，检查 `body_z > 0.9` 的触发频率。
- **侧向漂移过量**：虽然惩罚很小，若出现明显侧向漂移而不前进，说明主信号不强，需检查 `body_x_vel` 与 `body_y_vel` 的时间序列。