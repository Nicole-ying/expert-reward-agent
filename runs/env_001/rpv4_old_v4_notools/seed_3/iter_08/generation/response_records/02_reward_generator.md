# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励设计：基于 delta 距离的进展奖励 + 软着陆连续奖励 + 姿态速度约束 + 燃油惩罚
    """
    # 权重系数
    w_approach  = 1.0
    w_soft_land = 1.0
    w_stability = 0.05
    w_energy    = 0.05
    w_contact   = 0.1   # 双腿接触对齐小鼓励

    # 辅助常量
    speed_gate_alpha = 2.0
    land_coeff_v     = 2.0
    land_coeff_d     = 2.0

    # --- 当前和下一步状态提取 ---
    # 相对目标平台中心的位置
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]

    # 速度
    vx, vy = next_obs[2], next_obs[3]

    # 姿态
    angle      = next_obs[4]
    ang_vel    = next_obs[5]

    # 接触
    left_contact  = next_obs[6]
    right_contact = next_obs[7]

    # 动作：主引擎点燃是 2
    main_engine_used = 1.0 if action == 2 else 0.0

    # --- 计算欧氏距离 ---
    current_dist = (x**2 + y**2)**0.5
    next_dist    = (nx**2 + ny**2)**0.5

    # --- 1. 进展奖励（delta 距离 + 速度门控）---
    delta_dist = current_dist - next_dist   # 正值表示靠近目标
    abs_speed  = abs(vx) + abs(vy)
    speed_gate = 2.718281828 ** (-speed_gate_alpha * abs_speed)   # 用 e 的幂次，避免 import numpy
    approach_delta_gated = w_approach * delta_dist * speed_gate

    # --- 2. 软着陆连续奖励 ---
    # 双腿接触 + 低速 + 靠近中心 → 高奖励
    both_legs = left_contact * right_contact     # 0 或 1
    speed_factor = 2.718281828 ** (-land_coeff_v * abs_speed)
    dist_factor  = 2.718281828 ** (-land_coeff_d * next_dist)
    soft_landing_bonus = w_soft_land * both_legs * speed_factor * dist_factor

    # --- 3. 姿态/角速度稳定性惩罚 ---
    stability_penalty = -w_stability * (abs(angle) + abs(ang_vel))

    # --- 4. 主引擎燃油消耗惩罚 ---
    energy_penalty = -w_energy * main_engine_used

    # --- 5. 双腿接触对齐的小鼓励 ---
    contact_alignment = w_contact * both_legs   # 只有同时接触时才会得到小正向奖励

    # 总奖励
    total_reward = (
        approach_delta_gated
        + soft_landing_bonus
        + stability_penalty
        + energy_penalty
        + contact_alignment
    )

    components = {
        "approach_delta_gated": approach_delta_gated,
        "soft_landing_bonus":   soft_landing_bonus,
        "stability_penalty":    stability_penalty,
        "energy_penalty":       energy_penalty,
        "contact_alignment":    contact_alignment,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **navigation_goal_reaching**，子类型 **goal_approach_and_soft_contact**  
  任务核心：可控飞行器到达中央目标平台，并在双腿接触条件下以低速稳定停靠。

## selected reward roles
根据 `environment_card.md` 的 `reward_role_decomposition`，本次设计覆盖以下职责：

- **主学习信号：goal_proximity**  
  使用 delta 距离 + 速度门控的 `approach_delta_gated` 替代传统绝对值距离奖励。  
  变化：每一步奖励 **距离的减小量**（`current_dist - next_dist`），同时用速度门控压制高速靠近的收益，迫使 agent 在缓慢接近中获得更高奖励。  
  该设计完全依赖环境卡片中明确可用的 `obs[0:2]` 和 `next_obs[2:3]`。

- **软着陆成功：soft_contact_success**  
  采用 **乘积型软条件**（三个因子：双腿接触、低速、高接近度）构建连续奖励。  
  当 agent 双腿同时触地、速度极低且靠近平台中心时，获得接近 1.0 的稠密正向信号，引导最终稳定停靠。  
  该组件仅在条件满足时活跃，与主进展信号互补。

- **健康/稳定约束：stability_penalty**  
  惩罚身体角度（`next_obs[4]`）和角速度（`next_obs[5]`）的绝对值，防止姿态失控。

- **效率约束：energy_penalty**  
  每次使用主引擎（action==2）施加微小惩罚，抑制不必要的燃料消耗。

- **接触对齐辅助：contact_alignment**  
  小幅度奖励双腿同时接触（乘积），进一步鼓励对齐着陆，避免单腿弹跳。

## role_to_signal_mapping and formula operators
| Role | Signals used | Formula operator(s) |
|------|-------------|---------------------|
| goal_proximity | `obs[0:2]`, `next_obs[0:2]`, `next_obs[2:4]` | `improvement_delta` + `preview_conditioned_reward` 风格的 gate：`delta_dist * exp(-α·abs_speed)` |
| soft_contact_success | `next_obs[6]`, `next_obs[7]`, `next_obs[2:4]`, `next_obs[0:2]` | `joint_condition_proxy` (乘积形式) + `bounded_signal` (指数衰减) |
| stability_penalty | `next_obs[4]`, `next_obs[5]` | `absolute_value` 作为线性惩罚 |
| energy_efficiency | `action == 2` | `binary_penalty` |
| contact alignment | `next_obs[6]`, `next_obs[7]` | `product_of_contacts` 小正奖励 |

## excluded roles and reasons
- **terminal_success_reward / terminal_failure_penalty**：环境卡片显式声明 `explicit_success_flag_available: false` 且 `info` 为空，因此无法使用任何基于终止状态的硬奖励/惩罚。
- **fast_approach**：风险过高，可能诱导高速坠毁或失控，与 soft_contact 目标冲突，留到后续迭代。
- **action_smoothing**：离散动作空间，无平滑切换需求，且可能阻碍快速响应。
- **survival_balance_coherence**：本任务无平衡存活需求，主体不会因倾斜直接失败（除非撞击）。

## design hypothesis – why this structure is materially different
- 之前尝试的结构多依赖 **静态距离奖励**（如 `goal_proximity`）与分散的接触奖励。  
  本设计将距离信号改为 **delta进展 + 速度门控**，迫使 agent 在低速条件下逐步靠近目标，从根本上缓解了“高速冲撞依然得到高距离奖励”的风险。
- 软着陆奖励改用 **三个连续因子的乘积**（接触 × 低速 × 接近度），形成在理想着陆条件下唯一的、平滑递增的引导山丘，而不是二元事件奖励。
- 速度门控起到类似 `soft_health_gate` 的作用，在每步动态调节进展收益，鼓励减速靠近，这是之前设计未曾使用的机制。

## future iterations
- 可引入 **动态课程门控**，在训练后期增大 `soft_landing_bonus` 权重。
- 若 agent 出现单腿弹跳问题，可加入 **持续接触时长** 的奖励。
- 待训练稳定后，可降低 `stability_penalty` 权重，仅保留必要的姿态提醒。

## expected failure modes to monitor after training
- **低速徘徊不降落**：agent 可能在远离平台的地方低速震荡，需要检查 `approach_delta_gated` 和 `soft_landing_bonus` 的权重平衡。
- **双腿接触后提前奖励饱和导致不减速**：若 `soft_landing_bonus` 过早给予高值，agent 可能满足于有接触但速度较大的状态，需要调整系数 `land_coeff_v`。
- **过分保守导致难以抵达**：速度门控过严可能抑制必要机动，观察平均接近速度，必要时降低 `speed_gate_alpha`。
- **能量惩罚过大影响探索**：若 `main_engine_used` 比例过低，适当减小 `w_energy`。
