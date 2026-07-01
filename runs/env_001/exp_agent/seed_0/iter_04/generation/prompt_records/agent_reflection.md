# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你可以调用工具来搜索技法、查看骨架细节。

# 工具

- search_reward_design_knowledge(query)：搜索设计技法库。当你对某个症状不确定怎么修时，用自然语言搜索。如 "reward sparse trigger rate low"、"penalty dominating progress"、"oscillation near goal"。
- get_skeleton_detail(skeleton_name)：查看一个骨架的数学形态、原理和陷阱。当你想尝试不同的骨架时，先查看它的细节再决定。

# 怎么修订

你的工具箱里有三种层次的操作：

**层次 1：改系数。** 如果问题是量级不对（如惩罚太强），调系数。用 ratio_to_progress 判断：绝对值 > 0.5 且外部得分差 → 削 10 倍或改为距离门控。nonzero_rate < 2% 的正向组件 → 增大权重或放宽条件。

**层次 2：改表达式。** 如果系数调了几轮还是不行，考虑改变数学形式。例如：
- 二值 if 条件 → 连续乘积（每项用 max(0, 1-x/threshold)）
- 线性惩罚 → bounded 饱和（1/(1+kx)、tanh、exp(-x)）
- 全程生效 → 距离门控（只在靠近目标时生效）

**层次 3：换骨架 (rebuild)。** 如果当前骨架已经调了 2 轮以上、得分仍然远低于 target，不要继续在同一个骨架上微调。从 expert knowledge 中选一个数学形态不同的骨架重来。例如 progress_delta 不行 → 试试 bounded_proximity 或 potential_based_shaping。换骨架之前先调用 get_skeleton_detail 了解候选骨架。

# 常用技法速查

| 症状 | 可能解法 |
|------|---------|
| penalty 的 ratio_to_progress > 0.5，episode 短 | 削 10 倍；或 distance_gated（远处不罚） |
| bonus nonzero_rate < 2% | 换连续乘积形式；二值条件 → max(0,1-x/D) |
| progress 正常但 episode 长，不完成 | 添加 low-speed+low-angle shaping；尝试 bounded_proximity |
| generated_reward 高但 external 差 | 奖励黑客：检查哪个组件可被 exploit |
| 同一骨架 ≥2 轮无改善 | 调用工具查技法；如果还是没有新思路 → rebuild |

# 约束

- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 禁止发明未声明的 info 字段，禁止 import/eval/open。

# 输出

先写注释说明诊断和修改理由，再输出完整 Python 代码。
函数签名：def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
返回 (float(total_reward), components)，components 只放总公式中直接出现的变量。

```

## User Prompt

```markdown
# 环境契约
- function_signature: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
- allowed: obs[0..7], next_obs[0..7], action (0=noop,1=left,2=main,3=right), info (no reliable fields)
- forbidden: original_reward, official_reward, terminal_success_reward, terminal_failure_penalty


# 当前奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断与修改说明：
    # 问题：score=-115，100% early terminal。agent 在 progress_delta 引导下
    # 接近目标，但 crash 而非 soft landing。soft_landing_proxy 的 nonzero_rate=0.72%
    # 仍然太稀疏（乘积形式导致任一因子为零则整体为零）。
    #
    # 修改方案：
    # 1. 用 potential_based_shaping 替代 progress_delta_reward。
    #    势能函数 Φ = -(dist + 0.8*speed + 0.5*|angle|)，同时引导接近、减速、稳定。
    #    理论保证最优策略不变（Ng 1999），天然抗震荡。
    # 2. soft_landing_proxy 改为加权和形式（非乘积），提高 nonzero_rate。
    #    每个因子独立贡献梯度，不会因为某个条件不满足而完全消失。
    # 3. stability_penalty 保留角速度惩罚（距离门控），速度和角度已由 shaping 覆盖。
    # ============================================================
    
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    
    current_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    next_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    
    current_angle = abs(obs[4])
    next_angle = abs(next_obs[4])
    
    # ============================================================
    # 1. 主学习信号: potential_based_shaping
    #    势能 Φ = -(dist + 0.8*speed + 0.5*|angle|)
    #    shaping = γ * Φ(next) - Φ(obs), γ=0.99
    #    同时引导：接近目标 ↓、减速 ↓、姿态稳定 ↓
    # ============================================================
    gamma = 0.99
    phi_obs = -(current_dist + 0.8 * current_speed + 0.5 * current_angle)
    phi_next = -(next_dist + 0.8 * next_speed + 0.5 * next_angle)
    
    shaping_scale = 2.0
    potential_shaping = shaping_scale * (gamma * phi_next - phi_obs)

    # ============================================================
    # 2. 稳定约束: angular_vel_penalty（距离门控）
    #    速度和角度已由 shaping 覆盖，只保留角速度惩罚
    # ============================================================
    angular_vel = abs(next_obs[5])
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_dist / gate_radius)
    
    angular_vel_penalty_weight = 0.02
    angular_vel_penalty = -angular_vel_penalty_weight * angular_vel * distance_gate

    # ============================================================
    # 3. 任务完成proxy: soft_landing_proxy（加权和形式）
    #    用加权和替代乘积，每个因子独立贡献梯度
    #    提高 nonzero_rate，让 agent 逐步学会各个条件
    # ============================================================
    # 距离因子：dist < 0.8 时开始贡献
    dist_factor = max(0.0, 1.0 - next_dist / 0.8)
    # 速度因子：speed < 0.5 时开始贡献
    speed_factor = max(0.0, 1.0 - next_speed / 0.5)
    # 姿态角因子：angle < 0.3 时开始贡献
    angle_factor = max(0.0, 1.0 - next_angle / 0.3)
    # 接触因子
    contact_factor = min(1.0, (next_obs[6] + next_obs[7]) / 2.0)
    
    # 加权和（各因子权重不同，dist 最重要）
    landing_bonus_weight = 1.5
    soft_landing_proxy = landing_bonus_weight * (
        0.4 * dist_factor + 
        0.3 * speed_factor + 
        0.2 * angle_factor + 
        0.1 * contact_factor
    )

    # ============================================================
    # 4. 动作代价: energy_penalty（小权重）
    # ============================================================
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * engine_use

    # ============================================================
    # 总奖励
    # ============================================================
    total_reward = potential_shaping + angular_vel_penalty + soft_landing_proxy + energy_penalty

    components = {
        "potential_shaping": potential_shaping,
        "angular_vel_penalty": angular_vel_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈
# Training Feedback

## Training outcome
score=242.086713, len=441.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| angular_vel_penalty | -0.000592 | 0.000592 | 0.962926 | -0.000592 |
| energy_penalty | -0.007546 | 0.007546 | 0.754637 | -0.007546 |
| potential_shaping | 0.021351 | 0.030130 | 0.999998 | 0.021351 |
| soft_landing_proxy | 0.829484 | 0.829484 | 0.982930 | 0.829484 |
| total_reward | 0.842697 | 0.842805 | 1.000000 | 0.842697 |
| generated_reward | 0.842697 | 0.842805 | 1.000000 | 0.842697 |
| original_env_reward | -0.117375 | 1.916126 | 1.000000 | -0.117375 |

## Distribution
- score: mean=242.086713, min=216.675382, max=277.852200
- episode_length: mean=441.300000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -115.52 | -115.52 | 0.00 | 74.10 | energy_penalty=-0.002 progress_delta_reward=0.032 soft_landing_proxy=0.005 stability_penalty=-0.109 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -114.99 | -115.52 | 0.53 | 74.10 | energy_penalty=-0.001 progress_delta_reward=0.016 soft_landing_proxy=0.009 stability_penalty=-0.007 | no_meaningful_improvement |
| 3 | angular_vel_penalty + energy_penalty + potential_shaping + soft_landing_proxy | 242.09 | 242.09 | 0.00 | 441.30 | angular_vel_penalty=-0.001 energy_penalty=-0.008 potential_shaping=0.021 soft_landing_proxy=0.829 | target_solved_new_best |

```
