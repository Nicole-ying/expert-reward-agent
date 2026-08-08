# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- signal extraction (current and next) -----
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next, y_vel_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ----- distance to landing pad -----
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5

    # ----- component A: proximity progress (dense improvement delta) -----
    # Encourage moving closer to the pad; scale to make it a principal driver.
    w_prox = 0.15
    proximity_progress = w_prox * (dist_curr - dist_next)   # positive when distance shrinks

    # ----- component B: soft landing when legs contact the pad -----
    contact_any = 1.0 if (left_contact_next > 0.5 or right_contact_next > 0.5) else 0.0

    # thresholds for a safe landing (tune via experiment)
    vy_thresh = 0.2   # vertical speed very low
    vx_thresh = 0.2   # horizontal speed very low
    angle_thresh = 0.1 # radians, small tilt

    # bounded factors: each goes to 1 when the condition is perfectly met, 0 when threshold exceeded
    vy_factor = max(0.0, 1.0 - abs(y_vel_next) / vy_thresh)
    vx_factor = max(0.0, 1.0 - abs(x_vel_next) / vx_thresh)
    angle_factor = max(0.0, 1.0 - abs(angle_next) / angle_thresh)

    landing_quality = vy_factor * vx_factor * angle_factor  # joint condition proxy

    # The reward is given only when contact is active, thus encourages gentle touch-down.
    w_land = 0.8
    soft_landing_reward = contact_any * w_land * landing_quality

    # ----- component C: orientation stability (light penalty) -----
    w_ang = 0.05
    orientation_penalty = -w_ang * (angle_next**2 + ang_vel_next**2)

    # ----- assemble reward -----
    total_reward = proximity_progress + soft_landing_reward + orientation_penalty

    components = {
        'proximity_progress': proximity_progress,
        'soft_landing_reward': soft_landing_reward,
        'orientation_penalty': orientation_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  `navigation_goal_reaching` / `goal_approach_and_soft_contact`（2D 着陆器任务，discrete 控制）

- **selected reward roles**（按 environment_card 发布的 mandatory/conditional 职责）  
  1. `proximity_to_target` – 引导机体向着陆垫移动  
  2. `soft_landing_on_pad` – 接触垫子时必须低速、垂直，鼓励安全着陆  
  3. `orientation_stability` – 抑制过大倾斜和翻滚

  条件职责 `fuel_efficiency` 暂不加入 v1，留待后续迭代根据训练曲线按需引入（初期优先学会任务方向，避免压制必要的引擎探索）。

- **role_to_signal_mapping**  
  - `proximity_to_target` → 来自 `x_position`, `y_position` 的欧氏距离；使用 **improvement_delta** 算子 `dist_curr - dist_next` 作为稠密进展信号，每步均有梯度且能防止“停在高处不动”。  
  - `soft_landing_on_pad` → 利用 `left_contact`, `right_contact`, `y_velocity`, `x_velocity`, `body_angle`。采用 **joint_condition_proxy**（乘积形式）并乘以接触标志，只在腿接触时提供额外引导，鼓励同时满足低速度、小角度。  
  - `orientation_stability` → 对 `body_angle` 和 `angular_velocity` 施加 **quadratic_penalty**，轻量抑制姿态发散，不影响必要机动。

- **为何未使用 terminal_success_reward / terminal_failure_penalty**  
  environment_card 明确声明 `explicit_success_flag_available=false`、`explicit_failure_flag_available=false`，info 字典为空，不存在可安全判断成功或失败的信号。v1 不依赖任何硬编码终止事件奖励，避免错误发放。

- **excluded roles 及原因**  
  - `fuel_efficiency`：v1 优先建立稳固的主任务行为，发动机惩罚过早加入会抑制必要的探索与减速操作，留到 agent 稳定着陆后再引入。  
  - `time_pressure_soft`：缺少 episode 步数或时间信号，无法实现。  
  - `crash_penalty`、`out_of_bounds_penalty_early`、`explicit_success_bonus`：均因关键信号缺失或不可靠而被明确标记为 avoid_roles。

- **后续迭代计划**  
  1. 当 agent 在 v1 下开始稳定着陆后，可通过 `training_progress` 渐进加入 `fuel_efficiency`（小权重惩罚主引擎动作），进一步优化能耗。  
  2. 若有观察表明 agent 在下降段末期仍俯冲过快，可考虑引入 **soft_health_gate**（在高度很低且速度过大时衰减 proximity_progress），但此 gate 需要精确设计阈值，随后续实验结果确定。

- **训练后应重点观察的 failure modes**  
  - agent 在目标垫上方悬停而不着落（proximity_progress 已接近零，无动力下降，soft_landing 需要接触才能激活 → 可能需要微调 soft_landing 权重或引入条件）  
  - 高速撞击垫子但 episod 奖励仍较高（proximity_progress 很大，soft_landing 无法抵消 → 可考虑在 proximity_progress 中混入“速度上限”的 penalty 或后续添加 gate）  
  - 姿态大幅度摇摆导致无法稳定（orientation_penalty 权重若过小会失效）  
  - 引擎滥用但 v1 未惩罚（可接受，优先学会正确着陆再减燃料）