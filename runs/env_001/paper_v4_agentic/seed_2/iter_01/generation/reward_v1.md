# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for the 2D lander soft‑landing task.
    Components:
      - proximity_delta:  improvement in distance to target (core driving signal)
      - velocity_penalty: quadratic speed penalty gated by proximity (soft landing)
      - orientation_penalty: quadratic penalty on tilt and angular rate (safety)
    """
    # ── current state ──
    x_cur = obs[0]
    y_cur = obs[1]
    vx_cur = obs[2]
    vy_cur = obs[3]
    angle_cur = obs[4]
    angvel_cur = obs[5]

    # ── next state ──
    x_next = next_obs[0]
    y_next = next_obs[1]

    # ── distance to pad (target at 0, 0) ──
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # ── weights / thresholds ──
    w_prox = 1.0           # core progression weight
    w_vel  = 0.5           # velocity penalty weight
    w_ang  = 0.1           # orientation penalty weight
    proximity_threshold = 0.5   # distance below which we start caring about speed

    # ── 1. Proximity delta (improvement_delta) ──
    # positive when the lander reduces its distance to the target.
    proximity_delta = w_prox * (dist_cur - dist_next)

    # ── 2. Soft landing velocity penalty (quadratic_penalty + conditional_gating) ──
    # gate: 0 when far away, linear ramp to 1 when inside threshold.
    gate = max(0.0, 1.0 - dist_cur / proximity_threshold)
    speed_sq = vx_cur ** 2 + vy_cur ** 2
    velocity_penalty = -w_vel * speed_sq * gate

    # ── 3. Orientation stability (quadratic_penalty) ──
    orientation_penalty = -w_ang * (angle_cur ** 2 + angvel_cur ** 2)

    # ── Total reward ──
    total_reward = proximity_delta + velocity_penalty + orientation_penalty

    components = {
        "proximity_delta": proximity_delta,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
  着陆器需要从上方飞向固定垫并实现低速、小角度软着陆。

## selected reward roles (v1)
| role | objective | operator chosen | why selected |
|------|-----------|----------------|--------------|
| goal_proximity (mandatory) | 驱动 agent 缩小到目标垫的距离 | `improvement_delta` on L2 distance | 每步都有梯度，避免“站在原处也能得分”的停滞；正向 delta 鼓励持续接近 |
| soft_landing_velocity (mandatory) | 在接近着陆点时抑制水平和垂直速度 | `quadratic_penalty` × proximity gate | 仅当距离 < 0.5 时激活速度惩罚，避免过早抑制飞行；gate 线性 ramp 保证平滑 |
| orientation_stability (mandatory) | 保持车身水平，减小翻转风险 | `quadratic_penalty` | 全程轻量约束，防止 agent 为追求距离而剧烈倾斜 |

## role_to_signal_mapping
- **goal_proximity** ← `x_position`, `y_position` → derived L2 distance `dist = sqrt(x² + y²)` → `dist_curr - dist_next`
- **soft_landing_velocity** ← `x_velocity`, `y_velocity`, `distance` → gate = `max(0, 1 - dist/thresh)` → `-w * (vx² + vy²) * gate`
- **orientation_stability** ← `body_angle`, `angular_velocity` → `-w * (angle² + angvel²)`

## excluded roles & reasons
- **fuel_efficiency (conditional)**: v1 省略动作代价，避免抑制探索；agent 需先学会着陆姿态，后续迭代再加入轻微燃料惩罚。
- **soft_contact_bonus (conditional)**: 虽可取，但调参不当极易诱导“猛砸垫面”来获取接触奖励；计划在 v2 结合严格的速度/角度门槛后引入。
- **terminal_success_reward / terminal_failure_penalty**: 环境未提供显式成功/失败标志（`info` 为空），无法可靠发放离散奖励；`improvement_delta` 和速度惩罚已提供足够的稠密梯度。
- **dense_orientation_penalty_in_early_phase**: 初期随机推力可能引起大幅度摆动，但 `orientation_penalty` 权重 (0.1) 足够小，不会扼杀探索，故未额外抑制。

## quantities left for future iterations
- 燃油/动作效率惩罚（当前 agent 可以随意开引擎）
- 基于支撑腿接触的软着陆成功代理（需要条件门控 + 低速度/低角度验证）
- 飞出边界的硬惩罚（依赖距离监测，可在后续加入 `terminal_event` 风格惩罚）
- 动态课程：随 `training_progress` 逐渐收紧速度/角度约束

## failure modes to monitor after initial training
1. **悬停不敢降落**：若 episode 变长，主体停在远处且动作多为 no_engine，应考虑降低 `velocity_penalty` 权重或放大 `proximity_delta` 的引导。
2. **高速硬着陆**：撞击时垂直速度仍大 → 增大 `w_vel` 或缩小 `proximity_threshold`，使减速更早介入。
3. **水平飞出边界**：若 x 位置频繁越界，需在后继版本加入基于距离的边界惩罚。
4. **姿态抖振**：左右引擎频繁交替，`angular_velocity` 大 → 稍微提升 `w_ang` 或限制动作频率（v2 可加动作变化惩罚）。
5. **接触腿被利用**：目前未引入接触奖励，此风险暂不存在，但未来引入 `soft_contact_bonus` 时需严格条件。