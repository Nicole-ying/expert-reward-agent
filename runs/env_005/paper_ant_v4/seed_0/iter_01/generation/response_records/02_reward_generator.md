# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract next state signals ----
    h = next_obs[0]                # body_z (height)
    vx = next_obs[13]              # body_x_velocity (forward speed)
    quat_x = next_obs[2]           # quaternion x component
    quat_y = next_obs[3]           # quaternion y component

    # utility: body up projection (直立度)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---- 1) main learning signal: forward progress ----
    forward_reward = 1.0 * vx

    # ---- 2) survival health: hinge penalty near dangerous heights ----
    low_bound = 0.3
    high_bound = 0.8
    height_penalty_low = max(0.0, low_bound - h)
    height_penalty_high = max(0.0, h - high_bound)
    height_penalty = -20.0 * (height_penalty_low + height_penalty_high)

    # ---- 3) upright stability: soft quadratic penalty on deviation from upright ----
    upright_error = 1.0 - body_up_z
    upright_penalty = -0.5 * (upright_error ** 2)

    # ---- total reward ----
    total_reward = forward_reward + height_penalty + upright_penalty

    # ---- components dict ----
    components = {
        "forward_reward": forward_reward,
        "height_penalty": height_penalty,
        "upright_penalty": upright_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

**Selected task_family / dynamics_subtype**  
`locomotion_continuous_control` / `multi_legged_body_locomotion`（四足连续力矩控制，前向移动主目标）。

**Selected reward roles**  
1. **forward_progress**（mandatory） – 主学习信号，驱动机器人持续向前。  
2. **survival_health**（mandatory） – 安全高度约束，通过 softly‑hinged 惩罚避免高度靠近 0.2/1.0 终止边界。  
3. **upright_stability**（conditional）– 轻度姿态约束，鼓励躯干保持直立，减少侧倾/俯仰恶化。  

**Role‑to‑signal mapping**  
- forward_progress: `next_obs[13]`（body_x_velocity）  
- survival_health: `next_obs[0]`（body_z）  
- upright_stability: `body_up_z = 1 - 2*(next_obs[2]² + next_obs[3]²)`，导出自四元数

**Formula operators selected**  
- forward_progress: **dense_state_signal (positive linear)**  
  `w * vx`，简单直接，每步提供连续梯度；后续若出现低速平台可转为 quadratic。  
- survival_health: **dense_state_signal (hinge penalty)**  
  `-w * (max(0,low−h)+max(0,h−high))`，仅在高度进入危险区间时生效，避免 intra‑safe‑zone 的无谓惩罚。low=0.3, high=0.8 留有 0.1‑0.2 m 的预警带。  
- upright_stability: **quadratic_penalty**  
  `-w * (1−up_z)²`，给予平滑的微小惩罚，不影响正常步态摆动，但对大幅倾斜产生渐进压力。  

**Excluded roles (and why)**  
- `lateral_drift_penalty` – v1 不引入，避免过早限制策略探索侧向微调；若后续出现明显侧向漂移再加入。  
- `action_smoothness` – v1 不引入，效率/能耗优化留到后续迭代；过早加入可能压制探索动作幅度。  
- `goal_distance_progress`, `contact_force_management`, `energy_penalty_via_torque` – 必缺或无必要（详见 environment_card）。

**Why no terminal_success_reward / terminal_failure_penalty**  
- `explicit_success_flag_available = false`，无法定义“成功完成”。  
- `explicit_failure_flag_available` 虽为 true，但 `info` 完全不可用且禁止使用 `terminated` 信号；我们通过预判高度的 hinge 惩罚实现等效的“终止前引导”。

**Deferred roles for later iterations**  
- lateral drift suppression  
- action smoothness / torque magnitude minimization  
- dynamic curriculum or soft‑health gate（若出现“先冲后死”模式）

**Failure modes to watch after initial training**  
- 策略学会“原地踏步”，正向速度微弱 → 需逐渐增加 forward_progress 权重或改用凸化形式。  
- 频繁跌倒（高度 < 0.2）→ 强化高度惩罚（增大权重或收缩预警边界）。  
- 弹跳过猛（高度 > 1.0）→ 调整 high_bound 或额外引入 body_z_velocity 惩罚。
