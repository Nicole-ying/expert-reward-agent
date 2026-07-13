# Response Record

## 诊断与干预

1. **evidence**：landing_proxy active_rate仅6.0%（几何平均含contact_factor，飞行阶段恒为0导致整体塌缩），episode_sum_mean虽+229但集中爆发在少数接触步；distance_penalty均值-269全程支配梯度；15/20 terminated表明多数episode以某种终止结束，score范围[-206, +240]极宽说明部分episode碰巧着陆得分高但多数失败。上一轮将best的SUM改为几何平均+提权到8.0，得分21.66远未改善。

2. **behavior_diagnosis**：智能体在飞行阶段得不到任何正向引导——landing_proxy塌缩为0，仅有距离惩罚驱使其靠近目标，速度和姿态约束过弱（合计仅占1.2%+1.1%），导致智能体可能以高速、大倾角冲向目标区域，偶尔碰出接触获得burst reward，但无法稳定复现成功着陆。

3. **signal_completeness**：position_approaching（距离惩罚）✓，velocity_reduction（门控速度惩罚存在但过弱）⚠，orientation_stabilization（存在但过弱）⚠，safe_landing_confirmation存在但**不可达**——接触因子使整个几何平均在飞行阶段塌缩，无法提供着陆前的接近梯度。

4. **selected_level**：Level 2 — `product_to_noncollapsing_joint`。证据明确：几何平均中contact_factor在飞行阶段恒为零，导致product塌缩。需将连续因子（接近、低速、姿态）与离散接触事件解耦。

5. **selected_intervention**：仅修改landing_proxy结构。拆分为：(a) 三个连续因子的几何平均 `approach_quality = (prox_factor * vel_factor * angle_factor) ** (1/3)` 提供密集梯度；(b) 独立加性 `contact_bonus = contact_factor * approach_quality` 作为触地额外激励。系数 w_approach=5.0, w_contact=3.0 保持与当前8.0相近的最大值但分配梯度。

6. **falsifiable_hypothesis**：移除contact_factor后，approach_quality在飞行阶段不再塌缩，active_rate应从6%跃升至>50%，为智能体提供"同时满足接近、低速、端正"的连续梯度；contact_bonus在触地时按approach_quality缩放，激励最终着陆而不干扰接近学习。

7. **expected_next_round**：landing_proxy active_rate >50%，episode得分均值显著提升，score_range缩小（减少极端值），terminated率行为更一致，智能体展现出减速接近、姿态控制后触地的协调行为轨迹。

8. **main_risk**：approach_quality可能在目标附近形成局部最优——智能体悬停保持低速端正但拒绝触地；contact_bonus需3.0权重足够强以突破悬停惰性。若下一轮出现悬停行为（episode较长但score不高），需增强contact_bonus或添加时间压力。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: penalizes high speed when near target, gated by distance
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: geometric mean of 3 continuous factors (proximity, low speed,
      upright posture) providing dense approach gradient; plus additive contact bonus
      scaled by approach quality to incentivize final touchdown without collapsing
      the approach signal
    """
    # next_obs unpacking
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Euclidean distance to target
    dist = (x**2 + y**2) ** 0.5

    # Component A: distance penalty (core progress signal)
    w_dist = 1.0
    distance_penalty = -w_dist * dist

    # Component B: velocity penalty (damped by distance to target)
    w_vel = 0.2
    gate = 1.0 / (1.0 + dist)
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # Component C: orientation stabilization penalty
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # Component D: landing proxy with decoupled approach quality and contact bonus
    # Bounded factors for continuous dimensions
    prox_factor = max(0.0, 1.0 - dist / 2.5)
    speed = (vx**2 + vy**2) ** 0.5
    vel_factor = max(0.0, 1.0 - speed / 2.0)
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.5)

    # Geometric mean of 3 continuous factors: enforces joint satisfaction
    # of proximity, low speed, and upright posture during approach.
    # Unlike the 4-factor version, this does NOT collapse when airborne
    # (contact_factor was the zero-killer for 94% of steps).
    approach_quality = (prox_factor * vel_factor * angle_factor) ** (1.0 / 3.0)

    # Contact factor: 0.0 (none), 0.5 (single), 1.0 (double)
    contact_factor = (left_contact + right_contact) / 2.0

    # Contact bonus: additive extra reward when touching ground,
    # scaled by approach_quality so contact only pays off when
    # the vehicle is also close, slow, and upright.
    contact_bonus = contact_factor * approach_quality

    w_approach = 5.0
    w_contact = 3.0
    landing_proxy = w_approach * approach_quality + w_contact * contact_bonus

    # Total reward
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy

    # Component dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components
```
