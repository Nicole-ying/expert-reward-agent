# Response Record

`evidence`：Iter 2（best）score=-11.22，len=1000（truncated），progress=0.004，soft_landing_proxy=0.858；agent存活满episode但未着陆。Iter 3（当前）改为improvement-based后崩溃，score=-86.47，len=89，18/20 early terminal。对比证明improvement-based方向错误，state-value结构至少能到达目标附近，但存在hovering exploit。

`behavior_diagnosis`：agent学会在目标附近悬停来持续收取soft_landing_proxy奖励，而不真正接触着陆。因为landing_avg中speed_quality、angle_quality、angvel_quality即使没有contact也能贡献75%的最大均值（contact仅占1/4权重），导致hovering是局部最优策略。

`signal_completeness`：progress_reward提供接近目标的稠密引导（职责完备）。soft_landing_proxy提供着陆质量信号，但其**加法均值结构**允许各子条件互相补偿——无接触时仍可获得近端稳定性满分，缺失"必须接触才能获得显著奖励"的联合满足约束。缺少燃料效率信号和完成信号，但不是当前hovering的主因。

`selected_level`：Level 2 — 证据直接指向soft_landing_proxy的加法均值结构缺陷（独立子目标可互相补偿→hovering exploit），这是数学形态问题而非纯尺度问题。变换类型：`independent_to_joint`。

`selected_intervention`：唯一修改soft_landing_proxy组件，将contact从加法平均成分改为软性乘法门控。`(speed+angle+angvel+contact)/4` → `(speed+angle+angvel)/3 * (0.2 + 0.8*contact_quality)`。无接触时奖励降至20%地板，双足着地时完整释放。LANDING_SCALE保持2.0以维持着陆时的最大奖励量级。

`falsifiable_hypothesis`：contact门控使无接触hovering奖励从~1.5/步降至~0.4/步（约3.75×降幅），而着陆时仍为~2.0/步。这个相对差距应推动agent从hovering转向实际接触着陆，表现为episode在着陆成功时终止（len不再满1000），score改善。

`expected_next_round`：len从1000下降至中等范围（成功着陆终止），score从-11.22改善（负值减小或转正），early_terminal比例应低于iter 3但非零，soft_landing_proxy的episode_sum_mean可能因episode变短而下降但其per-step值在接触时应显著高于无接触时。

`main_risk`：20%地板值可能仍不足以在接触稀疏时提供足够梯度引导agent发现着陆动作；如果agent从未触发contact，landing信号持续过低可能导致其放弃接近目标区域。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取当前观测
    x, y = obs[0], obs[1]

    # 提取下一时刻观测
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    nl_contact = next_obs[6]
    nr_contact = next_obs[7]

    # 超参数
    PROGRESS_SCALE = 1.0
    CLIP_MAX = 1.0
    LANDING_SCALE = 2.0
    DIST_THRESH = 1.0
    VEL_THRESH = 0.3
    ANGLE_THRESH = 0.05
    ANGVEL_THRESH = 0.05

    # 主学习信号：向目标前进的稠密奖励
    dist_prev = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    delta_dist = dist_prev - dist_next
    delta_dist_clipped = max(-CLIP_MAX, min(CLIP_MAX, delta_dist))
    progress_reward = PROGRESS_SCALE * delta_dist_clipped

    # 近距门控因子
    proximity = max(0.0, 1.0 - dist_next / DIST_THRESH)

    # 稳定性子条件（不含contact）
    speed = (nvx**2 + nvy**2) ** 0.5
    speed_quality = max(0.0, 1.0 - speed / VEL_THRESH)
    angle_quality = max(0.0, 1.0 - abs(n_angle) / ANGLE_THRESH)
    angvel_quality = max(0.0, 1.0 - abs(n_angvel) / ANGVEL_THRESH)

    # contact作为软性乘法门控：无接触时奖励降至20%，双足着地时完整释放
    contact_quality = 0.5 * (nl_contact + nr_contact)
    contact_gate = 0.2 + 0.8 * contact_quality

    # 稳定性均值 + contact门控：联合满足结构，contact不可被速度和姿态完全补偿
    stability_avg = (speed_quality + angle_quality + angvel_quality) / 3.0
    soft_landing_proxy = LANDING_SCALE * proximity * stability_avg * contact_gate

    total_reward = progress_reward + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```
