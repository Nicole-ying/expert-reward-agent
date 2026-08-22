# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ====== 主学习信号：前进速度奖励 ======
    # 直接使用下一步的水平线速度，鼓励快速向前，后退自然为负
    forward_vel = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_vel

    # ====== 竖直/姿态稳定性惩罚 ======
    # 保持躯干直立、抑制大幅摇摆和垂直跳动
    hull_angle = obs[0]           # 躯干倾角
    hull_angvel = obs[1]          # 躯干角速度
    vertical_vel = obs[3]         # 垂直速度
    w_angle = 0.5
    w_angvel = 0.01
    w_vertical = 0.01
    stability_penalty = -(
        w_angle * hull_angle**2 +
        w_angvel * hull_angvel**2 +
        w_vertical * vertical_vel**2
    )

    # ====== 能量效率惩罚 ======
    # 降低关节力矩幅值，体现能耗目标
    w_energy = 0.001
    energy_penalty = -w_energy * (
        action[0]**2 + action[1]**2 +
        action[2]**2 + action[3]**2
    )

    total_reward = forward_reward + stability_penalty + energy_penalty
    components = {
        'forward_reward': forward_reward,
        'stability_penalty': stability_penalty,
        'energy_penalty': energy_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **任务画像**  
  `task_family`: locomotion_continuous_control  
  `dynamics_subtype`: planar_bipedal_gait  
  核心目标是快速前进，次要目标是节能与姿态稳定。环境不提供显式成功/失败标记（info 为空），因此所有奖励组件必须完全基于观测与动作信号。

- **选定的奖励职责与信号映射**  
  1. **forward_velocity_reward**（主学习信号）  
     - 信号：`next_obs[2]`（水平速度）  
     - 公式算子：`dense_state_signal` 的线性形式 `w * signal`  
     - 理由：水平速度直接反映任务核心进度，负值自然惩罚后退。
  2. **upright_stability_penalty**（安全/姿态约束）  
     - 信号：`obs[0]`（躯干倾角）、`obs[1]`（躯干角速度）、`obs[3]`（垂直速度）  
     - 公式算子：`quadratic_penalty`（三项独立二次型）  
     - 理由：抑制导致摔倒的主要前兆——过大倾角、剧烈摇摆和跳跃，但不阻碍正常步态中的小幅自然摆动。
  3. **energy_efficiency_penalty**（能耗约束）  
     - 信号：`action[0:4]`（关节力矩）  
     - 公式算子：`quadratic_penalty`（力矩平方和）  
     - 理由：任务明确将能量消耗列为重要次要指标；采用极小权重避免压制初期探索，同时提供节能梯度。

- **排除的职责与原因**  
  - `episodic_success_bonus`：info 为空，无任何成功/到达终点信号可用。  
  - `lidar_terrain_adaptation_reward`：雷达信息更适合作为状态输入而非奖励项，直接作为奖励可能扭曲任务目标。  
  - `alternating_contact_encouragement`：该步态约束在 v1 中可能干扰重心平衡学习，且依据环境卡片属于条件职责，留待后续迭代在稳定前行基础上加入。  
  - `imitation_of_masked_reward`：严格禁止，符合环境卡片约束。

- **不使用 terminal_success_reward / terminal_failure_penalty 的原因**  
  环境卡片中 `explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 false，并且 info 字段为空，无法在奖励函数内部可靠判断回合终止原因，因此不包含任何终端奖励或惩罚。

- **留待后续迭代的职责**  
  - `soft_health_gate`：当 agent 表现出“先冲后倒”的模式时，将躯干倾角等信息制作成乘性门控，在人形机器人学会稳定前行后进一步增强安全性。  
  - `alternating_contact_encouragement`：在基础步态形成后加入，用于精细化双足支撑相分布。  
  - `curriculum_weighting`：若后续发现早期探索与后期精细控制存在明显冲突，可引入训练进度相关的动态权重。

- **训练后需重点观察的失效模式**  
  1. **原地踏步或缓慢蠕行**：agent 为满足稳定性惩罚而不敢加速，可尝试适当提高 `w_forward` 或降低 `w_angle`。  
  2. **高速冲刺后跌倒**：主奖励驱动力过强，躯体倾角迅速恶化；应在后续考虑加入 soft_health_gate 提前衰减前进奖励。  
  3. **过度跳跃或猛摆**：垂直速度或角速度惩罚不足，可微调 `w_vertical` 和 `w_angvel`。  
  4. **能耗惩罚抑制有效步态**：若力量型步态无法出现，进一步减小 `w_energy` 或在早期训练中暂时关闭此项。