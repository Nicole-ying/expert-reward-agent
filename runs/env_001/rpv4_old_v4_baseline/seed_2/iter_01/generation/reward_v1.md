# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # distance to target (target at origin (0,0))
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # velocity components
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # orientation penalty: body angle and angular velocity
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # contact guidance: reward both legs touching, but only near the target and at low speed
    contact_val = float(next_obs[6] + next_obs[7])   # range 0.0 .. 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    velocity_ok = max(0.0, 1.0 - abs(vy) / 0.5)
    contact_guidance = 0.3 * contact_val * proximity_factor * velocity_ok

    # combine all components
    total_reward = (0.5 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 0.5 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`  
任务核心是控制飞行器靠近（0,0）目标平台，并在低速、水平姿态、双腿接触的条件下实现软着陆。

## selected reward roles
按照 `environment_card.md` 的 `reward_role_decomposition`，v1 选取了四个职责：
1. **goal_proximity** – 主学习信号，驱动飞行器向目标靠近。  
2. **velocity_management** – 稳定约束，抑制过大的水平/垂直速度，特别在接近地面时限制垂直速度，避免硬着陆。  
3. **orientation_stabilization** – 约束，保持机身接近水平并抑制角速度，防止因倾斜导致坠毁。  
4. **safe_contact** – 任务完成近似信号（soft proxy），鼓励双腿在目标附近且低速时接触，引导最终着陆姿态。

## role_to_signal_mapping
| role | signals used | mapping |
|------|--------------|---------|
| goal_proximity | `obs[0:,1]` & `next_obs[0:,1]` | `progress = dist_prev - dist_next`（improvement_delta） |
| velocity_management | `next_obs[2]` (vx), `next_obs[3]` (vy), `next_obs[1]` (y) | `quadratic_penalty` with altitude‑scaled factor for vertical speed |
| orientation_stabilization | `next_obs[4]` (angle), `next_obs[5]` (ang. vel.) | `absolute_penalty` on angle + `quadratic_penalty` on angular velocity |
| safe_contact | `next_obs[6], [7]` (leg contacts), `next_obs[0:,1]` (distance), `next_obs[3]` (vy) | `joint_condition_proxy` (product of proximity and low‑speed factors) |

## chosen formula operators
- **goal_proximity**: `improvement_delta`（`old_dist - new_dist`）。  
  理由：比静态距离奖励更能驱动 agent 持续靠近目标，避免在目标区停滞却不着陆时依然获得高额奖励。
- **velocity_management – vertical**: `quadratic_penalty` 乘以 `altitude_factor = 1/(1+10|y|)`。  
  理由：低空时惩罚放大，迫使软着陆；高空时允许较快下降，不压制探索。
- **velocity_management – horizontal**: 简单 `quadratic_penalty`，避免飞出视口。
- **orientation_stabilization**: `absolute_penalty` (角度) + `quadratic_penalty` (角速度)。  
  理由：连续梯度，抑制倾斜和快速旋转。
- **safe_contact**: `joint_condition_proxy` 乘积形式：`contact * proximity * (1 - |vy|/0.5)_+`。  
  理由：将稀疏腿接触信号连续化，只在满足空间和速度条件时才奖励，防止在任何地点假着陆获得奖励。

## excluded roles 及原因
- **sparse_success_reward**: `explicit_success_flag_available` = false，且 `info` 为空，无法可靠判断成功着陆。组合判断（低位、低速、双脚接触）仍有误判风险，v1 不做硬性终端奖励，避免奖励作弊。
- **engine_penalty**: 虽然燃料效率是次要目标，但 v1 阶段让 agent 先学会“怎么着陆”，效率优化留到后续迭代。过早惩罚引擎会压制探索和必要的姿态调整。
- **time_penalty / action_smoothness_penalty**：违背生存信号，易导致不安全急降；没有历史 action，无法实现平滑惩罚。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- 环境未提供 `info["success"]` 或类似标志，且终止原因无法从观测中唯一确定（`body_not_awake_or_settled` 可能由卡住、错误位置触发）。强行伪造终端奖励会严重误导策略。

## 哪些职责留到后续迭代
- **engine_efficiency**：在 agent 学会稳定着陆后再以极轻权重加入动作成本（如惩罚主引擎使用），优化燃料消耗。
- **time_to_land**：如果需要缩短 episode 长度，可在后期加入极小恒定负奖励（如‑0.005/step），但必须保证不引发危险行为。
- **soft_health_gate** 或 **preview_conditioned_reward**：如果出现“先冲向目标后摔死”或高度振荡的症时，可按需用 soft gate 切断主奖励；目前 v1 的 velocity_management 已提供初步防护。

## 训练后应该观察的 failure modes
| 预期失败模式 | 可能的表现 | 调优方向 |
|--------------|-----------|----------|
| 飞行器悬停在高空不下降 | 主奖励接近 0，y 位置在较高正值徘徊 | 增大 progress 权重或降低 velocity_penalty 的垂直分量在低空以外的部分 |
| 高速撞击平台 | 大量 crash 终止，vy 在低空仍很大 | 增强 vertical velocity penalty（提高 w_vy 或降低 altitude_factor 中的 10.0） |
| 水平飞出边界 | 因 x 坐标超出视口终止，x 速度大 | 增加 vel_horizontal_penalty 权重或添加 x 位置软约束 |
| 身体大幅度晃动致倾倒 | 接触时 body_angle 或 angular_velocity 仍高，导致坠毁 | 增加 orientation_penalty 权重，或在接触奖励中附加更严格的角度条件 |
| 在平台上方振荡不降落 | y 坐标在 0 附近上下波动且垂直速度正负反复 | 调整 velocity_penalty 的门控形式（考虑使用 hinge 仅在 vy 超过安全下降速度时惩罚），或放宽 contact_guidance 的 velocity_ok 条件 |
| 双腿接触但位置不在平台中央 | 触地时 x 坐标远离 0，但 reward 中接触奖励仍给出正值 | 收紧 proximity_factor 的阈值（当前 threshold=5.0），或增大 goal_proximity 权重 |