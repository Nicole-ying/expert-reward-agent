# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next observation
    x = next_obs[0]          # x position relative to target pad center
    y = next_obs[1]          # y position relative to target pad height
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity (negative = downward)
    angle = next_obs[4]      # body angle
    angvel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # A. Goal proximity shaping (mandatory)
    dist = (x**2 + y**2) ** 0.5
    proximity = 2.71828 ** (-dist)   # bounded positive: [0,1], peak at (0,0)

    # B. Soft landing condition (mandatory, task‑completion proxy)
    dist_factor = 2.71828 ** (-10.0 * dist)
    speed_sq = vx**2 + vy**2
    vel_factor = 2.71828 ** (-10.0 * speed_sq)
    angle_factor = 2.71828 ** (-10.0 * angle**2)
    angvel_factor = 2.71828 ** (-10.0 * angvel**2)
    contact_factor = left_contact + right_contact   # 0 when no contact, >0 otherwise
    soft_landing = dist_factor * vel_factor * angle_factor * angvel_factor * contact_factor

    # C. Energy efficiency (mandatory)
    energy_penalty = -0.01 if action != 0 else 0.0

    # D. Terminal velocity penalty (conditional, only near the pad)
    vel_penalty = 0.0
    if y < 0.05 and abs(x) < 0.1:
        # penalize excessively large downward speed
        if vy < -0.3:
            vel_penalty = -0.5 * max(0.0, -vy - 0.3)

    total_reward = 1.0 * proximity + 2.0 * soft_landing + energy_penalty + vel_penalty

    components = {
        "proximity": proximity,
        "soft_landing": soft_landing,
        "energy_penalty": energy_penalty,
        "terminal_velocity_penalty": vel_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **任务类型 / 动力学子类型**: `navigation_goal_reaching` / `goal_approach_and_soft_contact`。目标：到达中央目标垫并稳定停靠，最小推力。

- **选用的奖励职责与信号映射**:
  - `goal_proximity_shaping` (主学习信号) → 使用 `x, y` (next_obs[0], next_obs[1])，以指数衰减形式提供每步稠密梯度。
  - `soft_landing_condition` (任务完成近似) → 使用 `x, y, vx, vy, angle, angvel, left_contact, right_contact` 构造连续因子乘积，模拟成功着陆状态。
  - `energy_efficiency` → 使用 `action`，非零动作加小额惩罚。
  - `terminal_velocity_penalty` (条件约束) → 使用 `y, x, vy`，仅在极靠近垫面且已对准时惩罚过大负速度，防止硬着陆。

- **每个职责的公式算子**:
  - `proximity`: `exp(-dist)` – dense_state_signal 的 bounded 形式，在目标中心取得最大值，远离时平滑衰减。
  - `soft_landing`: 多个连续因子的乘积——`exp(-10·dist)`, `exp(-10·speed²)`, `exp(-10·angle²)`, `exp(-10·angvel²)`, 以及 `left_contact + right_contact`。符合 joint_condition_proxy 的设计思想；乘积保证所有条件共同满足才能获得显著奖励。
  - `energy_penalty`: 离散动作惩罚 `-0.01 * (action != 0)`。
  - `terminal_velocity_penalty`: hinge 惩罚 `max(0, -vy - 0.3)`，仅在 `y<0.05` 且 `|x|<0.1` 时激活。这避免了全局抑制下降，只在安全着陆窗口内约束速度。

- **排除的职责与原因**:
  - `explicit_success_reward`: 环境未提供 `info["success"]`，禁止使用。
  - `final_goal_sparse_reward`: 无可靠的显式成功标志，通过终止条件强行推断会引入 crash‑静止状态噪声。
  - `upright_orientation_bonus` (条件职责) 留到后续迭代：v1 中 soft_landing 的 angle/angvel 因子已间接鼓励姿态稳定，全局施加可能抑制必要的姿态调整探索。

- **为何未使用 terminal_success_reward / terminal_failure_penalty**: 环境卡片明确 `explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，所有终止相关信号不可用。

- **留到后续迭代的职责**:
  - `upright_orientation_bonus` 可在 agent 学会大体稳定着陆后添加，作为细化姿态的额外奖励。
  - `progress_penalty`（每步存活惩罚）暂不引入，避免迫使 agent 冒险硬着陆。

- **训练后应重点观察的 failure modes**:
  - 悬停在高处但不下落 → 检查 proximity 奖励是否在高处已饱和，可能需增强 `y` 梯度或增大 soft_landing 的门控范围。
  - 高速撞击导致 crash → terminal_velocity_penalty 参数可能太宽松，可降低速度阈值或加大惩罚系数。
  - 左右摇晃无法同时获得双接触 → 后续可考虑增加低姿态误差的轻量奖励，或在接近地面时对 `angle` 使用 hinge 约束。
  - 为节油过早关闭引擎而漂出视口 → 可能需增大 proximity 权重或添加水平边界软惩罚（后续迭代）。
