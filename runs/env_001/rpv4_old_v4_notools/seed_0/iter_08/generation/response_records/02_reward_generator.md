# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander: approach target pad, slow down, and settle
    with both support legs on the platform.

    Components:
      - progress (improvement delta on distance to target)
      - landing_gate_penalty (soft gate penalising speed/angle near platform)
      - contact_quality (bonus for dual leg contact with low speed & tilt)
    """
    # --- 1. Progress: distance reduction (improvement_delta) ---
    x_cur, y_cur = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    dist_cur = (x_cur**2 + y_cur**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = 10.0 * (dist_cur - dist_next)   # positive when approaching

    # --- 2. Soft landing gate: penalise speed & attitude only near ground ---
    y_threshold = 0.5               # how close to the platform the gate activates
    gate = max(0.0, 1.0 - abs(y_next) / y_threshold)  # 0 when far, ->1 when y≈0

    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    vel_penalty = vx**2 + vy**2
    angle_penalty = angle**2
    ang_vel_penalty = ang_vel**2

    landing_gate_penalty = -gate * (2.0 * vel_penalty + 1.0 * angle_penalty + 0.5 * ang_vel_penalty)

    # --- 3. Contact quality bonus: sparse reward for stable dual-leg contact ---
    left = next_obs[6]
    right = next_obs[7]
    contact_both = left * right               # 0 or 1

    # quality decays towards 0 when speed or tilt are large
    v_sq = vx**2 + vy**2
    angle_sq = angle**2
    v_scale = 0.1          # width of the exponential (tuneable)
    a_scale = 0.1
    quality = (2.718281828 ** (-v_sq / v_scale)) * (2.718281828 ** (-angle_sq / a_scale))

    contact_reward = contact_both * 20.0 * quality

    # --- assemble ---
    total = progress + landing_gate_penalty + contact_reward
    components = {
        'progress': progress,
        'landing_gate_penalty': landing_gate_penalty,
        'contact_quality': contact_reward
    }
    return total, components
```

# reward_v1 设计说明

- **任务族/子类**：`navigation_goal_reaching`，子类 `goal_approach_and_soft_contact`。
  核心是缩小水平与垂直偏差，最终以低速、双足接触、水平姿态稳定在目标平台。

- **选用的奖励职责**（取自环境卡片 `reward_role_decomposition`）：
  - `goal_approach_progress`（主学习信号）
  - `soft_contact_stability`（必需约束）
  - `orientation_keeping` 被整合进 `soft_landing_gate`，避免单独的全时惩罚。
  - `energy_efficiency_penalty` **排除**：v1 阶段应优先学会到达和软着陆，过早惩罚引擎动作会扼杀探索（前序实验已显示 `action_penalty` 组合得分停滞）。
  - `time_penalty`、`direct_success_bonus` **排除**：缺少时间测量和显式成功标志，且 info 为空。

- **信号映射与所用算子**：
  | 职责 | 使用的观测/动作 | 公式算子 | 理由 |
  |---|---|---|---|
  | `goal_approach_progress` | `obs[0], obs[1], next_obs[0], next_obs[1]` | `improvement_delta` (`dist_cur - dist_next`) | 每步都有梯度，agent 无法在远处静止，必须持续缩短距离。|
  | `soft_contact_stability` | `next_obs[1], next_obs[2:6]` | `soft_health_gate` （gate=`1 - |y|/threshold`） 与 `quadratic_penalty` | 只在接近地面时强约束速度和姿态，远处不干扰正常移动；二次惩罚保证平滑梯度。 |
  | `soft_contact_stability` (稀疏完成信号) | `next_obs[6], next_obs[7], next_obs[2:5]` | `joint_condition_proxy`（乘积 `contact_both * quality_exp`） | 双腿接触时给予指数衰减奖励，鼓励同时满足低速和水平姿态，提供明确的“成功着陆”引导。|

- **未使用 `terminal_success_reward` / `terminal_failure_penalty` 的原因**：
  - `explicit_success_flag_available = false`，`explicit_failure_flag_available = false`；
  - 环境 info 为空，无法可靠区分终止原因（crash vs 稳定着陆）。因此用 `improvement_delta` 和 `soft_landing_gate` 提供稠密引导，用稀疏但安全的 `contact_quality` 强化终态目标。

- **留到后续迭代的职责**：
  - `energy_efficiency_penalty`：在 agent 能稳定着陆后再加入，以避免早期行动抑制。
  - 更复杂的动态门控或课程调度（如根据训练进度逐渐收紧速度阈值）。

- **训练后应观察的失效模式**：
  - **agent 在远处悬停**：若平均距离不下降，需提高 `progress` 系数或检查是否 `landing_gate_penalty` 过早激活。
  - **高速撞击弹飞**：检查 `landing_gate_penalty` 的 `y_threshold` 和权重是否不足以在触地前减速；可考虑增大 `gate` 中的速度惩罚。
  - **单腿着地、翻滚**：`contact_quality` 中的 `contact_both` 条件会给出零奖励，但不足以阻止翻滚；需要 `landing_gate_penalty` 中的角度/角速度惩罚足够强。
  - **过度依赖 `contact_quality` 导致低速徘徊但不敢下落**：可进一步调整 `contact_reward` 的尺度（`v_scale`, `a_scale`），确保其在正常触地速度区间内能提供显著奖励。

该 v1 设计以 `progress` 为主要驱动力，`landing_gate_penalty` 提供接近地面时的安全减速与姿态稳定，`contact_quality` 作为最终“稳定停泊”的稀疏加强信号，整体覆盖了任务核心要求，同时避开了不可用信号和早期过度约束。
