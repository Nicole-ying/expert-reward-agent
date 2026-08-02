# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D lander goal-reaching task.
    Drives the agent to reach the target pad and settle gently.
    """
    # ---------- constants ----------
    PROGRESS_WEIGHT = 1.0
    LANDING_WEIGHT = 0.2
    ANGLE_PENALTY_WEIGHT = 0.01

    PROXIMITY_THRESHOLD = 0.5    # distance to start shaping soft landing
    ANGLE_THRESHOLD = 0.5        # rad
    VELOCITY_THRESHOLD = 0.5     # sum of absolute linear velocities

    # ---------- unpack observations ----------
    x_o, y_o, x_v_o, y_v_o, angle_o, _, left_o, right_o = tuple(obs)
    x_n, y_n, x_v_n, y_v_n, angle_n, _, left_n, right_n = tuple(next_obs)

    # ---------- 1) progress to target ----------
    R_obs = (x_o ** 2 + y_o ** 2) ** 0.5
    R_next = (x_n ** 2 + y_n ** 2) ** 0.5
    progress_reward = PROGRESS_WEIGHT * (R_obs - R_next)   # positive when getting closer

    # ---------- 2) soft landing incentive ----------
    proximity = max(0.0, 1.0 - R_next / PROXIMITY_THRESHOLD)
    angle_ok = max(0.0, 1.0 - abs(angle_n) / ANGLE_THRESHOLD)
    vel_ok = max(0.0, 1.0 - (abs(x_v_n) + abs(y_v_n)) / VELOCITY_THRESHOLD)
    contact_ok = left_n * right_n   # both support feet on the ground

    soft_landing = LANDING_WEIGHT * proximity * angle_ok * vel_ok * contact_ok

    # ---------- 3) light angular penalty ----------
    angle_penalty = -ANGLE_PENALTY_WEIGHT * (angle_n ** 2)

    # ---------- aggregate ----------
    total_reward = progress_reward + soft_landing + angle_penalty

    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "angle_penalty": angle_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 选定的任务画像与职责
- **task_family:** navigation_goal_reaching
- **dynamics_subtype:** goal_approach_and_soft_contact
- **强制职责（mandatory roles）:**
  1. **progress_to_target** – 每一步接近目标垫的欧氏距离变化
  2. **soft_landing_incentive** – 靠近目标时鼓励竖直姿态、低速、双支撑接触
- **额外健康约束（conditional but included）:** 轻量姿态二次惩罚，防止远离目标时过度倾斜导致失控。

## 职责‑信号映射与算子选择
| 职责 | 选用信号 | 公式算子 | 设计理由 |
|------|----------|----------|----------|
| progress_to_target | `x_position`, `y_position` (当前与下一步) | **improvement_delta** | 直接反映向目标的进展，比绝对距离奖励更稠密，不卡在“好状态停滞”的问题。 |
| soft_landing_incentive | `x_position`, `y_position`, `x_velocity`, `y_velocity`, `body_angle`, `left_support_contact`, `right_support_contact` (下一步) | **joint_condition_proxy**（乘积形式） | 用 proximity gate 限制在目标附近，然后要求角度、速度、接触三者同时满足；乘积强制 agent 必须完成全部着陆条件，避免悬停或部分满足奖励漏洞。 |
| angle_penalty | `body_angle` (下一步) | **quadratic_penalty** | 仅极小权重，全程抑制极端倾斜，降低翻滚风险，且不对主学习造成干扰。 |

**未使用信号说明：**  
- `angular_velocity` 未显式加入奖励，因为它已经隐含在姿态变化中；直接奖励角速度可能诱导高频抖动。  
- 动作本身未惩罚（效率/能耗）留到后续迭代，v1 以任务驱动力为首要。

## 排除的角色与原因
- **terminal_success_reward / terminal_failure_penalty:** 环境不提供显式成功/失败标志（`info` 为空），且终止信号不可直接获取。依赖乘积式软着陆信号和进展奖励即可引导稳定着陆，不需要硬编码终端事件。
- **efficiency / action cost:** 离散动作空间，首次迭代先学会到达与着陆，引擎使用量的优化放到 v2+。

## 后续迭代预留
- 引擎油耗代价（因任务要求节能）可在 agent 能稳定着陆后加入轻量 `action_efficiency` 项。
- 复杂轨迹优化（如时间最短）可结合 `training_progress` 动态调整权重，当前不启用。

## 预期失败模式及观察点
1. **速度冲击后弹跳** – 即使接触满足，若着陆时速度勉强低于阈值，乘积可能给正值但物理上仍会弹起。应关注 `soft_landing` 分量在着陆前后是否平滑，后续可收紧 `VELOCITY_THRESHOLD`。
2. **垫外稳定** – `progress_reward` 可能在边缘仍有小幅正值，导致 agent 停在垫子边缘不走。通过 `proximity` gate 和接触条件应能缓解。
3. **姿态振荡** – 若 `angle_penalty` 过小，agent 可能通过快速来回激振来获得距离进展。初期可监控 `angle_n` 的方差。