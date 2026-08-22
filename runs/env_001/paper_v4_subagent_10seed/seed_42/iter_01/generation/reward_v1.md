# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lunar-lander-style goal reaching task.
    """
    # ---- Unpack observations ----
    # obs: current state; next_obs: post-action state
    px0, py0 = obs[0], obs[1]
    px1, py1 = next_obs[0], next_obs[1]
    vx1, vy1 = next_obs[2], next_obs[3]
    angle1  = next_obs[4]
    angvel1 = next_obs[5]
    left_leg  = next_obs[6]
    right_leg = next_obs[7]

    # ---- 1. Progress to target: delta in Euclidean distance to (0,0) ----
    dist_prev = (px0**2 + py0**2) ** 0.5
    dist_next = (px1**2 + py1**2) ** 0.5
    progress_delta = dist_prev - dist_next   # positive when approaching

    # ---- 2. Orientation / stability soft constraints ----
    # Penalize large tilt and high angular velocity (use next_obs state)
    angle_penalty    = -0.01 * (angle1 ** 2)
    angvel_penalty   = -0.005 * (angvel1 ** 2)
    orientation_penalty = angle_penalty + angvel_penalty

    # ---- 3. Soft landing guidance (proximity-triggered proxy) ----
    # Activates only when the agent is close to the target pad.
    speed1 = (vx1**2 + vy1**2) ** 0.5
    proximity_threshold = 0.2          # tuned for the environment scale
    if dist_next < proximity_threshold:
        # contact factor: average of left/right leg contact (0..1)
        contact_factor = (left_leg + right_leg) / 2.0
        # speed smooth factor: 1 when speed=0, decays with higher speed
        speed_factor = 1.0 / (1.0 + 10.0 * speed1)
        soft_landing = contact_factor * speed_factor
    else:
        soft_landing = 0.0

    # ---- Combine components ----
    total_reward = (
        1.0 * progress_delta
        + 1.0 * orientation_penalty
        + 1.0 * soft_landing
    )

    components = {
        "progress_delta": progress_delta,
        "orientation_penalty": orientation_penalty,
        "soft_landing": soft_landing
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **任务族与动力学子类型**：`navigation_goal_reaching` + `goal_approach_and_soft_contact`。智能体需在离散动作下飞向目标并稳定停靠，省燃料为次要目标。

- **选用的奖励角色（roles）**：
  - `progress_to_target`（必修）：以每步距离缩减量 `dist_prev - dist_next` 作为主学习信号，提供连续梯度，避免静态悬停。
  - `stability_constraint`（条件性健康约束）：对躯干倾角 `body_angle` 和角速度施以轻量二次惩罚，防止翻滚、坠毁倾向。
  - `soft_landing_guidance`（任务完成近似信号）：当 `dist_next < 0.2` 时奖励低速度和双腿接触，将着陆行为连续化，补足缺少显式成功标志的问题。

- **角色-信号映射**：
  - 距离/位置 → `obs[:2]`, `next_obs[:2]`
  - 速度 → `next_obs[2:4]`
  - 姿态 → `next_obs[4]`, `next_obs[5]`
  - 接触 → `next_obs[6]`, `next_obs[7]`

- **所用公式算子**：
  - `improvement_delta`：距离缩减（正即奖励，负即惩罚）。
  - `quadratic_penalty`：倾角/角速度约束。
  - `bounded_signal` + 乘积：`1/(1+10*speed)` 和接触均值组合成软着陆 proxy，避免乘积塌缩并保持连续梯度。

- **排除的角色及原因**：
  - `fuel_efficiency`：v1 优先学习“飞向目标并稳定着陆”，动作代价会抑制探索，留待后续迭代。
  - `terminal_success_reward` / `terminal_failure_penalty`：`info` 为空且无显式标志，无法区分终止原因，因此不依赖终点事件。
  - `dynamic_curriculum` / `soft_health_gate`：v1 不需复杂的门控或课程，简单组件组合已覆盖核心引导和安全约束。

- **后续迭代建议**：
  - 加入动作效率惩罚（`action != 0` 罚）以降低燃料消耗；
  - 根据训练中 termination 模式引入 soft health gate 或 preview 因子；
  - 若 proximity 阈值周围出现盘旋/不降落行为，可改用分段课程或动态阈值。

- **预期训练后应观察的失效模式**：
  - 低速盘旋停在阈值外（soft_landing 未激活）；
  - 高速撞击目标（progress_delta 权重过高、缺少硬着陆压制）；
  - 单腿着陆后倾斜但不终止（接触 proxy 无惩罚）；
  - 姿态惩罚过大导致不敢点火，任务进度停滞。