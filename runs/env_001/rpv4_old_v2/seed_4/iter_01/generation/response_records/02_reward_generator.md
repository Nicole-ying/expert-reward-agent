# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 1. Target approach: improvement delta ----------
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = dist_curr - dist_next           # positive when agent moves closer
    w_approach = 5.0
    reward_approach = w_approach * progress

    # ---------- 2. Soft landing speed: distance‑gated velocity penalty ----------
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    sigma = 0.5                                # determines how sharply gate rises near target
    # Exponential gate: close to 1 only when dist_next is small
    gate = 2.718281828 ** (-dist_next**2 / sigma**2)
    w_speed = 2.0
    reward_speed = -w_speed * speed_next * gate

    # ---------- 3. Stable orientation: quadratic penalties on tilt and rotation ----------
    w_angle = 1.0
    w_angvel = 0.1
    reward_orientation = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # ---------- 4. Contact completion: dual‑leg contact gated by position and speed ----------
    left = left_contact_next
    right = right_contact_next
    beta_dist = 10.0
    beta_speed = 10.0
    # Soft proximity factors in [0,1] – near 1 only when distance/speed are near zero
    dist_factor = 1.0 / (1.0 + beta_dist * dist_next**2)
    speed_factor = 1.0 / (1.0 + beta_speed * speed_next**2)
    w_contact = 5.0
    reward_contact = w_contact * left * right * dist_factor * speed_factor

    total = reward_approach + reward_speed + reward_orientation + reward_contact

    components = {
        'approach_progress': reward_approach,
        'soft_landing_speed_penalty': reward_speed,
        'orientation_penalty': reward_orientation,
        'contact_completion': reward_contact,
    }
    return float(total), components
```

# reward_v1 设计说明

## 1. 任务画像与奖励职责选择

- **task_family**: navigation_goal_reaching  
- **dynamics_subtype**: goal_approach_and_soft_contact  
- **selected reward roles**（来自 environment_card.md 的 mandatory/conditional 分析）:
  1. **target_approach** – 驱动 agent 向目标位置移动
  2. **soft_landing_speed** – 在接近目标时抑制速度，防止高速撞击
  3. **stable_orientation** – 维持姿态竖直，抑制翻滚
  4. **contact_completion** – 奖励双支撑脚同时安全着地

这些角色直接对应任务核心目标：到达、低速接触、姿态稳定和成功着陆。  
v1 不包含次级目标（燃料/时间效率），待 agent 学会安全着陆后引入。

## 2. 职责‑信号映射与公式选择

| 职责                 | 观测信号                                               | 公式算子                | 具体形式（简化）                                                                 |
|----------------------|--------------------------------------------------------|-------------------------|----------------------------------------------------------------------------------|
| target_approach      | `x_position, y_position` (obs, next_obs)               | improvement_delta       | `w_approach * (dist_curr - dist_next)` 直接奖励距离的减小                        |
| soft_landing_speed   | `x_vel, y_vel, x_pos, y_pos` (next_obs)                | distance_gated_penalty  | `-w_speed * speed_next * exp(-dist_next² / sigma²)` 距离越近，速度惩罚越敏感     |
| stable_orientation   | `body_angle, angular_velocity` (next_obs)              | quadratic_penalty       | `-w_angle * angle² - w_angvel * angvel²` 轻量二次惩罚                           |
| contact_completion   | `left_contact, right_contact, x_pos, y_pos, x_vel, y_vel` (next_obs) | joint_condition_proxy   | `w_contact * left * right * (1/(1+βd d²)) * (1/(1+βs s²))` 仅在近距低速时激活 |

所有信号均来自观测空间，无使用 `info` 或 `original_reward`，符合接口契约。

## 3. 排除的职责与原因

- **fuel_efficiency**：v1 阶段以学会着陆为先，过早惩罚引擎动作可能抑制必要的机动；在 agent 完成稳定着陆后引入。
- **time_minimization**：环境不提供步数或计时信号，无法可靠实现，禁止使用。
- **explicit_collision_penalty**：缺少碰撞强度信号，且终止时环境已 respawn，奖励阶段无法可靠推断；不用 terminal_event 硬覆盖。
- **terminal_success_reward / terminal_failure_penalty**：`explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，且 `info` 为空，无法获取终止原因；依靠连续信号构造的 contact_completion 提供软完成引导。

## 4. 关键设计决策

- **主进展信号选用 improvement delta**：比简单的负距离更能驱动 agent 持续向目标移动，避免在近目标位置停滞。
- **距离门控的速度惩罚**：用 `exp(-d²/σ²)` 让惩罚在远离目标时几乎为零（不限制机动速度），仅在近距离时敏感，强制低冲击着陆。
- **contact_completion 使用连续乘积 proxy**：双脚接地 + 距离/速度因子确保只有安全着陆才获得高奖励，避免在远处或高速接触时 hack 奖励。
- **无终端事件硬奖励**：所有组件均为每步可用，提供稠密梯度。

## 5. 后续迭代方向

- 引入 `fuel_efficiency`（条件 idle 奖励），需在接近目标且速度足够低时启用。
- 考虑加入 `action_smoothness_penalty`（若未来提供动作历史 access）。
- 若训练出现“假着陆”（错误位置静止），增强 contact_completion 的位置条件约束。
- 可微调参数（sigma, beta_dist 等）以应对特定 failure modes。

## 6. 训练后应优先观察的 failure modes

| 现象                              | 诊断依据                                                                 | 可能干预                                       |
|-----------------------------------|--------------------------------------------------------------------------|------------------------------------------------|
| 高速撞击目标区域 crash             | episode 终止前 distance 极小但 speed 很大                                | 提高 w_speed 或减小 sigma，让 gate 更快激活    |
| 水平漂移出界                      | x_position 绝对值发散，未收敛到 0                                        | 增大 w_approach 或考虑轻量边界警告（在观测允许时）|
| 姿态倾斜导致非稳定着陆             | body_angle 较大，接触奖励频繁出现但 episode 因“静默”终止                 | 增加 w_angle，或让 orientation 惩罚与距离耦合   |
| 引擎过度使用                      | 大量主/姿态引擎动作，无明确任务进展                                       | 适时引入燃料效率组件（后续迭代）                |
| 异常“假着陆”在错误位置            | body_not_awake 终止但 distance 大，contact 奖励出现                      | 收紧 dist_factor 的门槛，或增大 beta_dist       |
