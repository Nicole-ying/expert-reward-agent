# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 基本观测提取 ----------
    hull_angle = obs[0]           # 身体倾角 (rad)
    horizontal_speed = obs[2]     # 质心水平速度

    # 只奖励正向移动 (假设正速度朝前)
    forward_vel = max(0.0, horizontal_speed)

    # ---------- 1. 主学习信号：gate‑controlled forward progress ----------
    beta = 2.0  # 姿态敏感度，beta 越大 gate 越早衰减
    gate = 1.0 / (1.0 + beta * abs(hull_angle))
    forward_reward = 1.0 * forward_vel * gate

    # ---------- 2. 稳定约束：hinge fall‑prevention penalty ----------
    # 只在身体倾角超过安全阈值 (0.6 rad ≈ 34°) 时才施加二次惩罚
    safe_angle = 0.6
    angle_excess = max(0.0, abs(hull_angle) - safe_angle)
    hinge_penalty = -0.5 * (angle_excess ** 2)

    # ---------- 3. 效率约束：轻量力矩代价 ----------
    action_cost = sum(action[i] ** 2 for i in range(4))
    energy_penalty = 0.01 * action_cost

    # ---------- 汇总 ----------
    total_reward = forward_reward + hinge_penalty - energy_penalty

    components = {
        "forward_reward": forward_reward,   # gate 调制后的前进奖励
        "hinge_penalty": hinge_penalty,     # 超出安全倾角时的二次惩罚 (≤0)
        "energy_penalty": -energy_penalty   # 力矩代价 (记录为负值)
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  `locomotion_continuous_control` · `bipedal_rough_terrain_locomotion`  
  双足机器人需在不规则地形上持续前进，核心目标是「走得远且不摔倒」。

- **selected reward roles**  
  1. *forward_progress*（主学习信号）  
  2. *fall_prevention*（稳定约束，hinge 形式）  
  3. *energy_penalty*（效率约束，权重极小）  

  不再单独使用 *balance_penalty* 或 *smooth_gait_penalty*，也没有使用 LIDAR 的任何奖励组件。

- **role_to_signal_mapping**  
  - forward_progress → `obs[2]` (horizontal_speed)，仅取正向速度  
  - fall_prevention → `obs[0]` (hull_angle)，**hinge 惩罚**仅当 `|hull_angle| > 0.6` 时生效  
  - energy_penalty → `action[0:4]`，二次惩罚 `sum(action_i²)`

- **每个 role 选择的 formula operator**  
  - forward_progress → **soft_health_gate**（健康状态恶化时压抑主奖励，而非独立惩罚）  
    - gate = `1 / (1 + beta * |hull_angle|)`，姿态偏差越大 gate 越接近 0  
  - fall_prevention → **dense_state_signal (hinge)**  
    - `hinge_penalty = -0.5 * max(0, |hull_angle| - 0.6)²`，只在危险区间提供严厉但短促的负梯度  
  - energy_penalty → **quadratic_penalty**  
    - `-0.01 * Σ action_i²`，系数极小仅起轻量抑制作用

- **excluded roles 及原因**  
  - *balance_penalty*（全时姿态惩罚）——容易压制探索，历史得分普遍为负  
  - *terrain_gate / roughness*——依赖 LIDAR 信号，LIDAR 只作为感知输入不应直接用于奖励，历史组合导致更低分  
  - *air_stability_penalty*——环境无法稳定判断腾空状态，且历史包含该组件的得分更差  
  - *terminal_success_reward / terminal_failure_penalty*——环境中 `explicit_success_flag_available = false`，且 `compute_reward` 不传入 `terminated` 信息，无法实现可靠的 terminal 事件  
  - *survival_bonus*——需要 `terminated` 标志，无法安全构造

- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**  
  环境没有显式成功/失败标志，且奖励函数接口不包含 `terminated` 参数，无法知晓 episode 是否结束。任何尝试从终止时刻的观测推断摔倒或到达终点的硬奖励都可能引入误判和不连续性，因此 v1 完全依赖稠密连续信号。

- **哪些职责留到后续迭代**  
  - 步态平滑性（smooth_gait_penalty）  
  - 基于 LIDAR 的预览门控（preview_conditioned_reward）  
  - 到达终点的稀疏完成奖励（当能通过训练长度/速度可靠推断时）  
  - 更复杂的动态权重或课程调整

- **训练后应该观察的 failure modes**  
  - 机器人原地摇摆却不前进：若 `forward_reward` 量级过小，可能增大 w_forward 或稍微降低 gate 的 beta  
  - 快速摔倒（早期终止）：hinge_penalty 也许太弱，需适当提高 w_hinge 或降低安全阈值  
  - 缓慢但高能耗行走：energy_penalty 太小，可逐步增加其系数（0.01 → 0.02）  
  - 机器人试图用小幅前倾获取高速度但长期不稳定：gate 使其在前倾时收获较少正向信号，预期能自然收敛到直立快速步态