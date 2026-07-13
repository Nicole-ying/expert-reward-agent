# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Implements V1 reward for navigation_goal_reaching (goal_approach_and_soft_contact).
    Uses a combination of proximity, velocity, orientation, and contact signals.
    """
    # Unpack next observation (state after action)
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = next_obs

    # --- Reward components ---
    # Goal proximity: Euclidean distance to target pad center (0,0)
    dist = (x**2 + y**2)**0.5
    goal_proximity = -30.0 * dist

    # Soft landing velocity: quadratic penalty on velocity magnitude
    # Encourages reducing both horizontal and vertical speed
    vel_penalty = -30.0 * (vx**2 + vy**2)

    # Orientation stability: quadratic penalty on body angle deviation from vertical
    angle_dev = angle**2
    orientation_penalty = -30.0 * angle_dev

    # Contact encouragement: bonus for each support leg contacting the pad
    contact_bonus = 10.0 * (left_contact + right_contact)

    # Combine
    total_reward = goal_proximity + vel_penalty + orientation_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "soft_landing_velocity": vel_penalty,
        "orientation_stability": orientation_penalty,
        "contact_encouragement": contact_bonus,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像确认
- `task_family`: `navigation_goal_reaching`
- `dynamics_subtype`: `goal_approach_and_soft_contact`
- 控制类型: 离散 (4 个动作)
- 形态: 2D 着陆器，带主引擎和左右姿态引擎，两条支撑腿

## 2. 选取的奖励职责
根据 `reward_role_decomposition` 中的 **mandatory roles** 选入以下四个职责，构成 v1 组件：
- `goal_proximity` (主学习信号)
- `soft_landing_velocity` (安全约束)
- `orientation_stability` (安全约束)
- `contact_encouragement` (辅助中期信号)

条件职责 `fuel_efficiency` 未在 v1 中使用；不使用的职责见 §4。

## 3. 职责 → 信号 → 公式算子
| 职责 | 使用的观测信号 | 公式算子 | 数学形式 |
|------|----------------|----------|----------|
| `goal_proximity` | `x_position`, `y_position` | `dense_state_signal` (penalty) | $-30\cdot\sqrt{x^2 + y^2}$ |
| `soft_landing_velocity` | `x_velocity`, `y_velocity` | `quadratic_penalty` | $-30\,(v_x^2 + v_y^2)$ |
| `orientation_stability` | `body_angle` | `quadratic_penalty` | $-30\,\theta^2$ |
| `contact_encouragement` | `left_support_contact`, `right_support_contact` | `binary_bonus` (正线性) | $+10\,(c_l + c_r)$ |

- 距离和速度惩罚使用相同的系数 (30.0) 以确保在主任务推进（下降）和安全约束之间取得平衡。
- 角度惩罚同样为 30.0，使用二次形式在偏离较小时提供较温和的梯度，在大倾斜时加大惩罚力度。
- 接触奖励系数 10.0 相对较小，提供额外的下降动机，但不会被反复弹跳利用（因为速度惩罚会抑制高频接触切换）。

## 4. 排除的职责及原因
- **`explicit_success_reward`**：环境无显式成功标志（`explicit_success_flag_available = false`），强行给予大正奖励风险极高，排除。
- **`progress_rew_from_original_reward`**：契约明确禁止使用 `original_reward`，排除。
- **`survival_bonus`**：任务非存活类，活得更久没有内在价值，反而可能推迟着陆，排除。
- **`fuel_efficiency`**（条件职责）：v1 优先让策略学会稳定接触和减速，过早加入燃料惩罚可能导致 agent 不敢使用引擎，留到后续迭代。

## 5. 为何没有 terminal 级信号
无显式 `terminal_success_reward` 和 `terminal_failure_penalty` 是因为环境不提供成功/失败标志，且终止原因（碰撞、出界、停稳）对 agent 而言不透明。v1 全部依赖稠密的连续信号，确保每一步都有可用的梯度。

## 6. 留到后续迭代的职责
- `fuel_efficiency`：在策略掌握基本着陆后引入，以优化燃料消耗。
- `joint_condition_proxy`：当 agent 能分别接近目标、减速、调正但难以同时满足时，可以用连续条件的乘积或几何平均提供一个“联合完成”的软奖励，进一步提高着陆精准度。
- 步数效率/时间惩罚：目前未引入，后续可考虑用 micro‑penalty 缩短 episode 长度。

## 7. 训练后需观察的 failure modes
| 现象 | 证据 | 可能调整方向 |
|------|------|--------------|
| 悬停不降 | `y_position` 不减小，`y_velocity ≈ 0`，episode 很长 | 加大 `goal_proximity` 权重或降低速度惩罚的底部死区 |
| 高速俯冲 | `y_velocity` 负值很大，crash 终止频繁 | 提升 `soft_landing_velocity` 权重，尤其对 y 速度 |
| 横向漂出视口 | `x_position` 发散，out-of-viewport 终止 | 加强水平距离/速度惩罚，或增加 `x_velocity` 单独罚项 |
| 反复弹跳无法 settled | `left/right_contact` 频繁跳变，`y_velocity` 正负交替 | 提高 `soft_landing_velocity` 权重，或增加对接触后速度的额外抑制 |
| 侧向倾斜着地失败 | `body_angle` 持续偏离，最终 crash | 增加 `orientation_stability` 权重或改用 hinge 形式只在超过安全阈值时重罚 |

这些调整方向可在 v2 中结合训练曲线进行针对性修改。
