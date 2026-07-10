# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 前进速度奖励（只奖励正向移动）
    hor_vel = next_obs[2]
    forward_reward = 2.0 * max(0.0, hor_vel)

    # 2. 平衡/稳定惩罚：身体倾角与角速度的平方惩罚
    angle = next_obs[0]
    ang_vel = next_obs[1]
    stability_penalty = -2.0 * (angle ** 2) - 0.5 * (ang_vel ** 2)

    # 3. 节能惩罚：关节力矩平方和
    energy_penalty = -0.005 * (action[0] ** 2 + action[1] ** 2 +
                               action[2] ** 2 + action[3] ** 2)

    total_reward = forward_reward + stability_penalty + energy_penalty

    components = {
        "forward_velocity": forward_reward,
        "stability": stability_penalty,
        "energy": energy_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  `locomotion_continuous_control` / `planar_bipedal_gait`。  
  双足平面身体，通过连续髋/膝力矩控制在不平地形上持续行走。

- **selected reward roles**  
  遵循 `environment_card.md` 中 `reward_role_decomposition` 的 mandatory roles：

  1. **forward_velocity_reward**（主学习信号）  
     - 使用信号：`next_obs[2]`（horizontal_velocity）。  
     - 公式算子：`dense_state_signal` 的变体 — `w * max(0, signal)`，只奖励向前分量，避免负速度惩罚干扰探索。  
     - 权重 2.0，使该组件成为核心驱动力。

  2. **stability_balance_penalty**（安全/健康约束）  
     - 使用信号：`next_obs[0]`（hull_angle）和 `next_obs[1]`（hull_angular_velocity）。  
     - 公式算子：`quadratic_penalty` — 倾角与角速度的平方惩罚，引导直立姿态。  
     - 权重：角度 2.0，角速度 0.5，适度约束而不压制动作尝试。

  3. **energy_efficiency_penalty**（效率/动作代价，小权重）  
     - 使用信号：`action[0:4]`（关节力矩）。  
     - 公式算子：`quadratic_penalty` — 力矩平方和。  
     - 权重极低（0.005），避免早期学习阶段抑制动作探索，同时为后续节能训练提供方向梯度。

- **excluded roles**  
  - `terminal_success_reward`：环境未提供显式成功标记（`explicit_success_flag_available=false`），且终止原因无法从 `terminated` 中可靠分离，因此不能使用。  
  - `terminal_failure_penalty`：同理，缺乏显式失败标记。  
  - `gait_contact_pattern`（conditional）：条件要求“初步站稳或移步后”使用，v1 阶段策略尚在探索基础移动，引入接触模式可能干扰早期学习，留待后续迭代。  
  - `terrain_aware_caution`（conditional）：需要从 10 维激光雷达中提取地形陡峭度，v1 先聚焦于本体信号，未来可加入基于雷达的 penalty/gate。

- **为何不使用 terminal_success_reward / terminal_failure_penalty**  
  环境 `info` 字典为空，无法获知回合是因到达终点还是摔倒终止，基于终止类型的激励不可用。所有奖励必须由步内的连续状态和动作构造。

- **留到后续迭代的职责**  
  1. `gait_contact_pattern`：当 agent 能稳定前进后，引入交替触地奖励（如惩罚双腿同时离地），可改善步态效率。  
  2. `terrain_aware_caution`：在基础移动稳定后，利用激光雷达信号感知前方陡坡，提前调整姿态。  
  3. 更强的 `energy` 权重或动态课程：训练后期可逐步加大节能权重或使用 `curriculum_weighting`，实现真正的低能耗运动。  
  4. 更精细的 `stability` 策略（如 `soft_health_gate` 替代独立惩罚），避免惩罚项与前进目标直接抵消。

- **训练后应观察的失效模式**  
  - `in_place_hopping`：agent 通过高频小跳维持存活但不前进，此时 `forward_reward` 较低，需检查奖励-速度关系。  
  - `velocity_burst_then_fall`：为了短时间内获得高前进奖励而激烈动作，随后摔倒，需关注稳定性惩罚是否充分。  
  - `jerky_high_energy_gait`：虽能前进，但动作幅度过大，能耗惩罚失灵，可能需要后期调高能量权重。  
  - `tilt_oscillation`：身体反复倾斜回正，若 `stability` 惩罚对倾角的瞬态反应不够，可能长期存在。
