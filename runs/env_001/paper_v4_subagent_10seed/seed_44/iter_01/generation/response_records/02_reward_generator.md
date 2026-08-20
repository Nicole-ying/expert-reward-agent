# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations: both obs and next_obs are 8-dim vectors
    x,        y        = obs[0], obs[1]
    vx,       vy       = obs[2], obs[3]
    angle,    ang_vel  = obs[4], obs[5]
    # next_obs
    nx,       ny       = next_obs[0], next_obs[1]
    nvx,      nvy      = next_obs[2], next_obs[3]
    nangle,   nang_vel = next_obs[4], next_obs[5]
    lcon,     rcon     = next_obs[6], next_obs[7]  # contact flags at next state

    # ----- potential function (smaller values are better) -----
    def potential(px, py, pvx, pvy, pa):
        dist = (px**2 + py**2) ** 0.5
        vel  = (pvx**2 + pvy**2) ** 0.5
        return -(2.0 * dist + 1.0 * vel + 1.0 * abs(pa))

    # Main progress signal: improvement in potential
    pot_old = potential(x, y, vx, vy, angle)
    pot_new = potential(nx, ny, nvx, nvy, nangle)
    progress = pot_new - pot_old
    # Scale factor can be tuned, keep raw for now. Usually we want reward per step in range ~1.0
    main_progress = progress   # expected range roughly [-?..+?], but typical improvement gives ~0.1-1.0

    # ----- fuel efficiency (action cost) -----
    # action 0 = no engine, 1/2/3 = use engine
    fuel_penalty = -0.02 if action != 0 else 0.0

    # ----- extreme tilt hinge (hard safety) -----
    tilt = abs(nangle)
    tilt_limit = 0.5   # radians, strongly tilted
    if tilt > tilt_limit:
        extreme_tilt_penalty = -0.5 * (tilt - tilt_limit)
    else:
        extreme_tilt_penalty = 0.0

    # ----- soft contact encouragement (only when close to target) -----
    dist_to_target = (nx**2 + ny**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + dist_to_target)   # close → 1, far → 0
    contact_bonus = 0.2 * lcon * rcon * proximity_factor

    # ----- total reward -----
    total_reward = main_progress + fuel_penalty + extreme_tilt_penalty + contact_bonus

    components = {
        "potential_delta": main_progress,
        "fuel_penalty": fuel_penalty,
        "extreme_tilt_penalty": extreme_tilt_penalty,
        "stable_contact_bonus": contact_bonus
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与角色选择

- **task_family**: navigation_goal_reaching  
- **dynamics_subtype**: goal_approach_and_soft_contact  
- **selected reward roles**（基于环境卡片中的任务目标与可用信号）:
  - **主进展信号**：`potential_based_shaping` — 同时驱动位置靠近、速度减小、姿态归正，每步提供梯度。
  - **安全约束**：`extreme_tilt_penalty` — 避免身体大角度倾斜导致的 crash 类失败，使用 hinge 仅在危险姿态时激活。
  - **任务完成软激励**：`stable_contact_bonus` — 双脚接触并接近目标时给予小奖励，鼓励最终着陆。
  - **效率/成本**：`fuel_penalty` — 轻量惩罚引擎使用，呼应“最小化燃料消耗”的次要目标。

## 2. 职责-信号映射

| 角色                   | 信号来源        | 具体观测维度                 |
|------------------------|-----------------|------------------------------|
| 主进展（potential）    | obs/next_obs    | x,y (0,1), vx,vy (2,3), angle (4) |
| 姿态安全约束（hinge）  | next_obs        | body_angle (4)               |
| 接触鼓励               | next_obs        | left_contact (6), right_contact (7), 同时用位置 (0,1) 计算 proximity |
| 燃料代价               | action          | action id (0~3)              |

所有信号均在环境卡片声明的可用范围内，未使用任何 info 字段。

## 3. 公式算子选择

- **主进展** → *potential_based_shaping* (3.3)  
  构造 potential = -(2×dist + 1×vel + 1×|angle|)，取其步间增量作为奖励。此形式比直接二次惩罚更有利于避免“静止不动也能得小奖励”的问题，迫使 agent 不断改善距离、速度和姿态。选择系数 2.0 / 1.0 / 1.0 基于粗略量级平衡（dist 通常 ≤2.8，vel 通常 ≤2，angle ≤π），使三项贡献可比。
- **姿态 hinge** → *dense_state_signal (hinge)* (3.1)  
  阈值 0.5 rad 为安全倾斜上限；超出部分线性惩罚，低于阈值时不施加压力，避免正常调节时被惩罚。
- **双脚接触** → *joint_condition_proxy + bounded_signal* (3.8, 3.9)  
  乘积形式 `lcon * rcon * proximity` 确保只有两脚同时接触且靠近目标时才获得奖励，proximity 用 1/(1+dist) 压缩距离影响。
- **燃料成本** → *action_efficiency* 的离散版 (3.7)  
  恒定小负值，系数 0.02 约为 main progress 步均量级的 2~5%，不压制探索。

## 4. 排除的 role 及原因

- **terminal_success_reward**：explicit_success_flag_available=false，无法安全使用。
- **terminal_failure_penalty**：explicit_failure_flag_available=false，且终止原因不直接暴露；改用 hinge 持续防御极端姿态。
- **复杂的 action_smoothness / dynamic curriculum / heavy gated_reward**：v1 阶段不需要，且缺少 previous action 等历史信号。
- **额外的角度/速度二次惩罚**：potential 已包含角度和速度的改进驱动力，二次额外惩罚会造成信号冲突（原则4）。

## 5. 设计理由（为什么没有使用 terminal_success / failure）

info 字典为空，无法获取 `success`、`failure` 或 `termination_reason`。环境中的所有终止（成功 settled 或 crash）均无显式标签，因此不能依赖终点硬奖励。取而代之，本设计使用 (a) potential_delta 持续推动接近完美着陆状态，(b) hinge 防止危险姿态，(c) contact_bonus 引出最终双脚接触，间接引导至 settled 状态。

## 6. 留待后续迭代的职责

- **更精细的 energy efficiency**：当前只有简单的动作计数惩罚，后续可引入主引擎/姿态引擎的差异化代价或连续推力幅值（若动作空间变为连续）。
- **基于前方信息的 preview**：目前缺少前方地形等信号，若以后扩展为三维或加入距离传感器可加入 preview 门控。
- **动态 curriculum**：在 agent 学会基本接近后再加大速度/姿态权重或减小燃料惩罚。
- **严格 terminal failure 惩罚**：如果能通过后续环境更新获得 termination_reason，可加入基于事件的失败惩罚。

## 7. 训练后应重点观察的 failure modes

- **速度冲击后弹开**：agent 可能惯性过大，即使 potential 包含速度项，也可能先加速冲近再猛烈弹回。需观察 episode 内速度曲线。
- **悬浮不动**：若 fuel penalty 过强或主进展 delta 收敛到零，agent 可能选择少动、挂起。应监测 engine 使用频率和 main progress 分量值。
- **反复触碰-离地刷 contact bonus**：contact_bonus 可能被利用，需检查双脚接触的持续性及是否伴随位置大幅度抖动。
- **极端姿态未完全遏制**：hinge 只在 >0.5 rad 时才惩罚，若 agent 学会贴着 0.5 rad 飞行并频繁触发 reset，考虑降低阈值或引入角速度约束。
