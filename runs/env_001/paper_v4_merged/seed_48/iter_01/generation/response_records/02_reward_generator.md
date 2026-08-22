# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs / next_obs : [x, y, vx, vy, angle, ang_vel, left_contact, right_contact]
    # goal is at origin (0,0); x,y relative to target pad
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    delta_distance = dist_old - dist_new  # positive when approaching

    # Soft landing progress: encourage closeness, low speed, low angle
    near_goal = 1.0 / (1.0 + 5.0 * dist_new)
    speed_sq = next_obs[2]**2 + next_obs[3]**2
    low_speed = 1.0 / (1.0 + 10.0 * speed_sq)
    abs_angle = abs(next_obs[4])
    low_angle = 1.0 / (1.0 + 20.0 * abs_angle)
    soft_progress = near_goal * low_speed * low_angle

    # Health constraint: penalize large body angle (hinge)
    angle_threshold = 0.4
    angle_penalty = max(0.0, abs_angle - angle_threshold)

    # Engine usage penalty: penalize any thrust action (discrete actions 1,2,3)
    engine_penalty = 1.0 if action != 0 else 0.0

    # Weights (tuned for scale balance)
    w_dist = 10.0
    w_soft = 2.0
    w_health = 0.5
    w_engine = 0.01

    total = (w_dist * delta_distance +
             w_soft * soft_progress -
             w_health * angle_penalty -
             w_engine * engine_penalty)

    components = {
        'distance_delta': w_dist * delta_distance,
        'soft_landing_progress': w_soft * soft_progress,
        'angle_penalty': -w_health * angle_penalty,
        'engine_penalty': -w_engine * engine_penalty,
    }
    return float(total), components
```

# reward_v1 设计说明

## 任务画像
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`（离散推力飞行器，双腿接触着陆）
- **核心目标**: 驱使飞行器快速到达并稳定停靠在中央目标平台；附属目标为减少引擎使用、保持姿态安全。

## 所选奖励角色及其信号映射与公式算子

| 角色 | 信号来源 | 公式算子 | 原因 |
|------|---------|----------|------|
| **distance_delta** (主进展) | `obs[0], obs[1], next_obs[0], next_obs[1]` | `improvement_delta` | 鼓励每步向目标靠近，避免悬停；用距离差提供稠密梯度。 |
| **soft_landing_progress** (软着陆近似) | `next_obs[0], next_obs[1], next_obs[2:4], next_obs[4]` | `joint_condition_proxy`（乘积形式，各因子 bounded） | 无 explicit success flag，用近距×低速×小角度乘积构建连续激励，引导 agent 在接近时自然减速、调平并轻柔着陆。 |
| **angle_penalty** (健康约束) | `next_obs[4]` | `dense_state_signal` (hinge) | 只在机体倾角超过安全阈值（0.4 rad）时惩罚，防止翻转/失控，同时不压制早期正常调整。 |
| **engine_penalty** (效率代价) | `action` | `constant_penalty` | 弱惩罚任何引擎使用（动作 1/2/3），鼓励节约推力；权重极小（0.01），避免拖慢主任务。 |

## 排除的角色及原因
- **soft_landing_terminal_bonus**（原 mandatory）：环境不提供 episode 终止标志（`compute_reward` 无 `done` 参数），无法在最后一步施加 terminal bonus，因此转化为每步的 `soft_landing_progress` 连续信号。
- **terminal_failure_penalty** / **terminal_success_reward**：`explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，info 为空，不具备可靠事件判决条件。
- **constant_survival_bonus**：禁止角色，与“尽快到达”冲突，会制造悬停陷阱。
- **angular_velocity_smoothness**：v1 不做额外约束；若后期观测到高频姿态抖动再考虑加入轻微阻尼。

## 未在 v1 使用的职责（留待后续迭代）
- **显式 success/failure 终端奖励**：待环境接口扩展或训练统计可推断终止原因后再加入。
- **出界/碰撞硬惩罚**：当前无法可靠推断，但 angle hinge 间接抑制失控。
- **双腿接触的必要性约束**：目前由 `soft_landing_progress` 的位置与速度条件近似；后续可按需加入双腿接触乘积因子或单独奖励。

## 训练后应观察的 failure modes
- **高速坠毁**：若距离奖励主导，agent 可能全速冲向平台，速度过快导致 crash。应关注 `soft_landing_progress` 的活跃程度，必要时调整其权重或引入速度上限 hinge。
- **悬停不降**：如果 agent 停在目标上空但距离不再变化，检查 `delta_distance` 是否提供足够梯度（应为进步信号，不会在静止时奖励）；若出现应增大 `soft_landing_progress` 权重。
- **单腿接触歪斜着陆**：偶尔只有一侧支撑腿接触，可能导致后续翻倒。后续可加入双腿同时接触的条件因子或额外小奖励。
- **姿态震荡**：若 `angle` 持续超限但 `angle_penalty` 未有效抑制，可降低阈值或在此后引入 `angular_velocity` 惩罚。
- **过度保守不动**：初始引擎惩罚过大可能导致 agent 不敢点火而被风吹出界；当前惩罚极小，不太可能发生，但仍需观察早期动作多样性。

---

*设计严格遵循 environment_card 与 expert_reward_context 的约束，仅使用声明的观测维度，不依赖 info 或原始奖励。*
