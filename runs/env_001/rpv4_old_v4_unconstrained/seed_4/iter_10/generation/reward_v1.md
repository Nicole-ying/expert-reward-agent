# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # observations: [x, y, vx, vy, angle, angular_vel, left_contact, right_contact]
    x, y, vx, vy, angle, av, left_c, right_c = obs
    nx, ny, nvx, nvy, nangle, nav, nleft_c, nright_c = next_obs

    # --- A: progress based on distance squared to target ---
    d_sq_obs = x**2 + y**2
    d_sq_next = nx**2 + ny**2
    progress = 2.0 * (d_sq_obs - d_sq_next)   # positive if moving toward (0,0)

    # --- B: light stability penalty on attitude and velocities ---
    stability = (
        -0.02 * abs(nangle)
        - 0.01 * abs(nav)
        - 0.02 * abs(nvx)
        - 0.01 * abs(nvy)
    )

    # --- C: soft contact bonus (two‑leg landing, near ground, low vertical speed) ---
    proximity = 1.0 / (1.0 + abs(ny))
    low_speed_gate = 1.0 / (1.0 + abs(nvy))
    both_legs = min(nleft_c, nright_c)          # 1.0 only when both legs touch
    contact_bonus = 1.0 * both_legs * proximity * low_speed_gate

    # --- D: tiny fuel/action cost ---
    action_penalty = -0.01 if action != 0 else 0.0

    total_reward = progress + stability + contact_bonus + action_penalty

    components = {
        "progress_delta": progress,
        "stability_penalty": stability,
        "contact_bonus": contact_bonus,
        "action_penalty": action_penalty
    }
    return total_reward, components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family**：`navigation_goal_reaching`
- **dynamics_subtype**：`goal_approach_and_soft_contact`
- 核心是引导飞行器到达原点目标垫并实现低速、竖直、双腿触地的稳定着陆。

## selected reward roles
- **mandatory**：`approach_and_landing_shaping`（通过 `progress_delta`、`stability_penalty`、`contact_bonus` 三个组件共同实现）
- **conditional**：`action_penalty`（微小权重，鼓励节省燃料但不阻碍飞行）

## role_to_signal_mapping 及所选 formula operators
| 角色 | 使用的信号 | 公式算子 | 说明 |
|------|-----------|---------|------|
| 主进展 | `x`, `y` (obs 和 next_obs) | **improvement_delta**（平方距离改善） | 奖励向目标移动的每步进展，防止悬停 |
| 姿态/速度约束 | `angle`, `angular_velocity`, `vx`, `vy` (next_obs) | **dense_state_signal**（线性绝对值惩罚） | 轻量抑制过大姿态和速度，保持飞行稳定 |
| 接触奖励 | `left_contact`, `right_contact`, `y`, `vy` (next_obs) | **joint_condition_proxy**（乘积门控） | 仅在接近地面且低速时奖励双腿接触，鼓励安全着陆 |
| 动作效率 | `action` | **action_efficiency**（离散惩罚） | 微小引擎使用代价 |

## excluded roles 及原因
- **`terminal_success_bonus`**：环境 card 中 `explicit_success_flag_available = false` 且 `info` 为空，无法获得终止事件。
- **`info_based_failure_penalty`**：同样，`info` 不提供任何失败标志，无法实现。
- 这两个职责完全由密集的 shaping 和约束覆盖，不需要硬终端信号。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
因为任务没有显式的成功/失败 flag，`info` 字典为空。所有成功/失败条件必须通过观测状态（位置、速度、姿态、接触）推断，我们的奖励结构已经用连续信号表达了这些条件，不需要稀疏的终端事件。

## 哪些职责留到后续迭代
- **更强的速度/能耗优化**（`action_penalty` 目前极小，若后期燃料消耗过高可逐渐放大）
- **动态课程或多阶段门控**（`training_progress` 目前未使用，后续可加入当 agent 学会基本靠近后再强调减速和竖直）
- **更严格的安全边界**（如 explicit hinge on x 超出视口，当前未硬编码，因为 progress delta 已隐含驱动向中心）

## 初始训练后应观察的 failure modes
- **悬停或缓慢下降**：若 `progress_delta` 权重过大且速度惩罚抑制垂直运动，可能导致 agent 不敢下降。应检查 horizontal/vertical velocity 是否被过度惩罚。
- **高速撞击**：若 `contact_bonus` 的门控（`low_speed_gate`）不够强，agent 可能学习到快速下落获取接触奖励。需观察接触发生时的 `vy` 分布。
- **姿态振荡**：`stability_penalty` 权重过小可能导致翻滚，过大则抑制姿态调整。观察 `angle` 和 `angular_velocity` 的统计。
- **过度使用引擎**：`action_penalty` 极小，燃料消耗可能仍然很高，若成为问题可在后续迭代中适度增加，或引入频率相关惩罚。