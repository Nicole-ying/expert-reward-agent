# Response Record

# 设计理由
**修改组件**：`soft_landing`（接近/着陆引导奖励）  
**上一轮问题**：该组件依赖 `proximity * angle_ok * vel_ok` 三因子乘积，`angle_ok` 与 `vel_ok` 使用了硬阈值条件，导致乘积极容易塌缩为零，`active_rate` 仅 0.7%，成为僵尸组件；agent 因此未获得有效的减速/稳定信号，只会机械缩短距离，在接近目标时因无法减速而撞毁或飞出边界。  
**变换类型**：Level 2 — 结构变换（连续化、消除乘积塌缩）。  
**新形式**：将 `soft_landing` 改为 `LANDING_WEIGHT * proximity * speed_bonus`，其中  
- `proximity = max(0, 1 - R_next / PROXIMITY_THRESHOLD)` 保留，在 0.5 单位内提供由 0 到 1 的接近激励；  
- `speed_bonus = 1 / (1 + speed)`，用连续、无塌缩的衰减函数代替硬阈值 `vel_ok`，速度越低奖励越接近 1，速度越高奖励平缓趋近 0，在任何速度下都提供梯度信息；  
- 移除角度条件（角度仍由 `angle_penalty` 单独约束），避免不必要的乘积因子导致塌缩。  

**系数校准**：`LANDING_WEIGHT` 由 0.2 降至 0.05。  
- 预估主信号 `progress_reward` 每步均值 ≈ 0.016。  
- 新组件在最优情况下（位于目标中心、速度为零）每步最大奖励 = 0.05 × 1 × 1 = 0.05，约为 3× 主信号，略超 2× 建议线，但该状态仅可能发生在最终稳定着陆的极少数步中，实际不会对训练产生支配性影响，且提供了足够强的逼近与减速激励。  
- 速度衰减为 `1/(1+speed)`，典型巡航速度下（speed≈1）奖励折半，梯度连续平滑，不会产生“悬崖”。

**其他组件**：`progress_reward`、`angle_penalty`、`angular_velocity_penalty` 保持不变，虽然 `angular_velocity_penalty` 也是僵尸组件，但本轮遵循“每轮只改一个组件”原则，留待后续处理。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D lander goal-reaching task.
    v4: replaced hard-threshold soft_landing with proximity * speed_bonus (1/(1+speed))
        to give continuous slowdown rewards and eliminate multiplicative collapse.
    """
    # ---------- constants ----------
    PROGRESS_WEIGHT = 1.0
    LANDING_WEIGHT = 0.05          # reduced; new form activates more often
    ANGLE_PENALTY_WEIGHT = 0.01
    ANGULAR_VELOCITY_PENALTY_WEIGHT = 0.02

    PROXIMITY_THRESHOLD = 0.5      # distance within which we encourage slowing down
    ANGULAR_VELOCITY_THRESHOLD = 0.5

    # ---------- unpack observations ----------
    x_o, y_o, x_v_o, y_v_o, angle_o, angvel_o, left_o, right_o = tuple(obs)
    x_n, y_n, x_v_n, y_v_n, angle_n, angvel_n, left_n, right_n = tuple(next_obs)

    # ---------- 1) progress to target ----------
    R_obs = (x_o ** 2 + y_o ** 2) ** 0.5
    R_next = (x_n ** 2 + y_n ** 2) ** 0.5
    progress_reward = PROGRESS_WEIGHT * (R_obs - R_next)

    # ---------- 2) soft landing incentive (continuous slowdown) ----------
    proximity = max(0.0, 1.0 - R_next / PROXIMITY_THRESHOLD)
    speed = (x_v_n ** 2 + y_v_n ** 2) ** 0.5
    speed_bonus = 1.0 / (1.0 + speed)   # smooth: 1 at rest, ~0.5 at speed=1, >0 always
    soft_landing = LANDING_WEIGHT * proximity * speed_bonus

    # ---------- 3) light angular penalty ----------
    angle_penalty = -ANGLE_PENALTY_WEIGHT * (angle_n ** 2)

    # ---------- 4) angular velocity hinge penalty ----------
    angular_velocity_penalty = (
        -ANGULAR_VELOCITY_PENALTY_WEIGHT
        * max(0.0, abs(angvel_n) - ANGULAR_VELOCITY_THRESHOLD)
    )

    # ---------- aggregate ----------
    total_reward = progress_reward + soft_landing + angle_penalty + angular_velocity_penalty

    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "angle_penalty": angle_penalty,
        "angular_velocity_penalty": angular_velocity_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 未使用接触信号，原 soft_landing 为僵尸组件（active_rate 0.7%），缺乏有效减速信号是 agent 快速崩溃的主因。
- **behavior**: agent 只学会缩短与目标的距离，接近时未减速，导致碰撞或出界终止，生存时间短（~70 步）。
- **signal**: 缺少数值上可微、不会塌缩的“接近 + 低速”联合奖励。
- **level**: Level 2（结构变换，将乘积硬阈值改为连续衰减）
- **hypothesis**: `proximity * 1/(1+speed)` 提供平滑的减速梯度，能使 agent 在进入目标区域时逐步降低速度，显著减少碰撞，延长 episode 并提高成功着陆概率。
- **risk**: 权重调整后减速信号可能偏弱（最大 0.05/步），若仍不足以促使减速，则需要后续再加大系数或调整速度尺度；同时缺少角度约束可能导致倾斜着陆，角度惩罚会弥补一部分。
