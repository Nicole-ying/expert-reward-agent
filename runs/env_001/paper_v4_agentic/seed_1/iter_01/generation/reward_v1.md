# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ================== main progress：减少到目标垫的欧氏距离 ==================
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_old - dist_new                    # 距离缩小为正奖励
    w_progress = 1.0

    # ================== 姿态/角速度稳定约束（hinge 软惩罚）===================
    body_angle = abs(obs[4])
    ang_vel = abs(obs[5]) if abs(obs[5]) > 1e-6 else 0.0

    angle_penalty = max(0.0, body_angle - 0.3)        # 角度超过0.3 rad 才惩罚
    ang_vel_penalty = max(0.0, ang_vel - 1.0)         # 角速度超过1.0 rad/s 才惩罚

    w_angle = 0.05
    w_ang_vel = 0.02

    # ================== 登陆完成软代理（joint_condition_proxy）=================
    # 用 next_obs 判断着陆条件
    proximity = 1.0 / (1.0 + 5.0 * dist_new)           # 越近越接近1
    leg_contact = next_obs[6] * next_obs[7]            # 双腿都接触=1.0，否则0
    speed = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    speed_factor = 1.0 / (1.0 + 5.0 * speed)           # 低速接近1

    # 几何平均缓解乘积塌缩
    proxy = (proximity * leg_contact * speed_factor) ** (1.0/3.0)
    w_proxy = 0.3

    # ================== 总奖励 ==================
    total_reward = (w_progress * progress
                    - w_angle * angle_penalty
                    - w_ang_vel * ang_vel_penalty
                    + w_proxy * proxy)

    components = {
        "progress": w_progress * progress,
        "angle_penalty": -w_angle * angle_penalty,
        "ang_vel_penalty": -w_ang_vel * ang_vel_penalty,
        "landing_proxy": w_proxy * proxy
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`

## selected reward roles (按职责预算)
| role | 预算类型 | 说明 |
|---|---|---|
| `progress` | 主学习信号（1个） | 鼓励每一步缩小与目标垫的欧氏距离，提供密集、连续的进展梯度 |
| `stability_constraints` | 稳定/安全约束（2个） | 通过 hinge 惩罚抑制过大机体倾角和过快自旋，避免失控坠毁 |
| `landing_proxy` | 任务完成近似信号（1个） | 使用多个连续条件（近垫、双腿接触、低速）构造软成功代理，弥补无显式 success flag 的缺失 |

## role‑to‑signal mapping
- **progress**: `distance = (x² + y²)^{1/2}`，利用 `obs[0], obs[1]` 与 `next_obs[0], next_obs[1]`
- **stability**: `body_angle` → `obs[4]`（机体倾角）；`angular_velocity` → `obs[5]`
- **landing_proxy**: 距离 → `next_obs[0], next_obs[1]`；腿接触 → `next_obs[6], next_obs[7]`；速度 → `next_obs[2], next_obs[3]`

## 各 role 使用的 formula operator
- `progress` → **improvement_delta**：`dist_old - dist_new`，期望距离持续减小。
- `stability` → **hinge**（`max(0, value - threshold)`）：只在角度或角速度超出安全边界时施加惩罚，避免在安全区域内持续压迫探索。
- `landing_proxy` → **joint_condition_proxy**（geometry mean of bounded factors）：`proximity`, `leg_contact`, `speed_factor` 三个 bounded 因子取几何平均，防止单一为0时整体塌缩。

## excluded roles 及原因
- **terminal_success_reward / terminal_failure_penalty**：`explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，且 info 为空，无法可靠判断最后一步；改用 dense `landing_proxy` 平滑引导。
- **action_efficiency / fuel cost**：v1 阶段先学会“抵达并稳定”的核心行为，燃油优化后续迭代再加入。
- **action_smoothness**：缺少上一时刻 action，无法实现。
- **soft_health_gate**：当前阶段 terminated 模式未知，过早使用可能抑制有效探索；先用 hinge 稳定姿态，后续若出现“先冲后死”再考虑门控。
- **potential_based_shaping**：本例存在天然的进展度量（距离），`improvement_delta` 已足够表达“朝目标前进”，无需额外构造势函数。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境卡片明确标记 `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，且 `info` 返回空字典。虽然可从观测推断成功或失败的状态，但无法在训练循环中可靠识别“最后一步”来发放一次性终端奖励，因此将任务完成和失败防止的目标转化为每步的稠密信号（`landing_proxy` 和 `stability_constraints`）。

## 后续迭代预留的职责
- 燃油消耗/引擎使用惩罚（`action_efficiency`）：利用离散动作 0‑3 构造轻量负奖励。
- 动态门控或课程式着陆奖励：当 agent 初步掌握下降后，再引入更严格的姿态‑速度联合条件。
- 针对悬停/不动的惩罚：若 agent 学会原地漂浮而不下降，可加入“长时间无进展”的负反馈。

## 训练后应重点观察的 failure modes
- **原地悬停**：距离不缩小，progress≈0，但 landing_proxy 也为 0，agent 可能停滞。
- **快速俯冲后撞击**：hinge 可能来不及在高速下提供足够制动力，导致 crash，需关注终止分布。
- **只下降不横向对齐**：progress 由 Euclidean 距离驱动，若 agent 忽略 x 方向校正，会在 ±x 方向偏离目标垫；观察 episode 末端的 x 位置偏差。
- **姿态震荡**：角度/角速度惩罚不足时可能出现高频摆动，需看角度惩罚的 active_rate 和最终着陆时的 body_angle。