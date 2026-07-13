# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyper-parameters (v1) ----------
    PROGRESS_WEIGHT = 2.0          # weight for distance reduction
    SOFT_LAND_WEIGHT = 0.5         # weight for speed penalty near pad
    LANDING_THRESHOLD = 1.0        # distance below which speed penalty activates
    ANGLE_WEIGHT = 1.0             # weight for body angle error
    ANGVEL_WEIGHT = 0.5            # weight for angular velocity penalty

    # ---------- compute current and next distance to target pad ----------
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5

    # ---------- main learning signal: approach progress ----------
    progress = dist_curr - dist_next   # positive when moving closer
    reward_progress = PROGRESS_WEIGHT * progress

    # ---------- stability constraint 1: soft-landing regularization ----------
    speed = abs(next_obs[2]) + abs(next_obs[3])
    landing_gate = max(0.0, 1.0 - dist_next / LANDING_THRESHOLD)
    soft_land_penalty = -SOFT_LAND_WEIGHT * speed * landing_gate

    # ---------- stability constraint 2: orientation stability ----------
    angle = next_obs[4]
    angvel = next_obs[5]
    orientation_penalty = -ANGLE_WEIGHT * (angle**2) - ANGVEL_WEIGHT * (angvel**2)

    # ---------- total reward ----------
    total_reward = reward_progress + soft_land_penalty + orientation_penalty

    components = {
        "progress": reward_progress,
        "soft_land_penalty": soft_land_penalty,
        "orientation_penalty": orientation_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与路线选择
- **任务家族**：`navigation_goal_reaching`
- **动力学子类型**：`goal_approach_and_soft_contact`（2D 着陆器，离散推力，通过两支撑脚接触目标垫完成软着陆）
- **主要目标**：到达目标垫并形成稳定接触（双足着垫、近零速度）
- **次要目标**：节省燃料（v1 不纳入核心奖励）

## 2. 所选奖励职责及理由
按照 `environment_card.md` 的 `reward_role_decomposition`，v1 选择以下职责并映射为三个组件：

| 职责角色 | 对应组件 | 选择的 formula operator | 理由 |
|---|---|---|---|
| `approach_progress`（主学习信号） | `progress` | **improvement_delta** （`dist_curr - dist_next`） | 每步提供直接梯度，鼓励向目标垫移动；比静态负距离更明确地奖励“靠近行为”，可避免悬停静止。 |
| `soft_landing_regularization`（稳定约束） | `soft_land_penalty` | **dense_state_signal (hinge 门控乘法)** | 结合距离门 `max(0, 1 − dist_next/threshold)` 只在接近目标垫时惩罚水平与垂直合速度，防止高速撞击。远距离不惩罚速度，保留探索机动能力。 |
| `orientation_stability`（稳定约束） | `orientation_penalty` | **quadratic_penalty** （角度平方 + 角速度平方） | 惩罚机体倾斜和剧烈旋转，引导直立姿态，确保安全接触。权重较小，避免抑制姿态发动机的必要使用。 |

## 3. 职责-信号映射验证
- `progress` 使用 `obs[0:2]` 与 `next_obs[0:2]` 计算笛卡尔距离，符合 `approach_progress` 的可用信号 `x_position, y_position`。
- `soft_land_penalty` 使用 `next_obs[2:4]`（速度）与 `next_obs[0:2]`（构造距离门），匹配 `soft_landing_regularization` 的 `x_velocity, y_velocity + gating(distance)` 映射。
- `orientation_penalty` 使用 `next_obs[4]`（角度）与 `next_obs[5]`（角速度），完全对应 `orientation_stability` 的 `body_angle, angular_velocity`。

所有信号均在观测向量的声明索引范围内，无需使用禁止信号。

## 4. 排除的职责及原因
| 排除的职责 | 原因 |
|---|---|
| `fuel_efficiency_penalty` | v1 优先学会安全着陆，过早惩罚发动机使用会严重抑制探索。留在后续迭代。 |
| `terminal_bonus_for_stable_contact` | 环境中无显式成功标志（`explicit_success_flag_available=false`），终止信息不可靠；且 v1 已用连续成型信号驱动接近与软着陆，稀疏终端奖励不会显著改善学习，反而可能引入错误引导。 |
| 边界越界惩罚 | 虽可防止飞出视口，但维持简单优先：距离减少信号本身会抑制横向发散，姿态稳定性限制角速度，二者协作可降低出界概率。强边界惩罚留至 v2 若仍出现高越界率时添加。 |
| `dynamic_curriculum` / `action_smoothness` / 强门控 | 缺乏历史动作与训练进度依赖，违反 v1 阶段设计原则。 |

## 5. 公式算子选择细节
- **progress（improvement_delta）**：`dist_curr - dist_next` 直接奖励距离减小量，避免线性负距离造成的悬停无梯度问题；若 agent 后退会被惩罚，驱动持续靠近目标垫。
- **soft_land_penalty（hinge × speed）**：`gate = max(0, 1 − dist_next / threshold)` 提供平滑的、仅在大约 1 单位距离内生效的速度惩罚。当 `dist_next < LANDING_THRESHOLD` 时，gate > 0，惩罚随距离减小而线性增强，促使末段减速。
- **orientation_penalty（quadratic）**：平方惩罚对小幅偏离容忍度高，对大幅偏离惩罚强，适合引导直立而不扼杀调整姿态的动作。将角度与角速度分开加权，可独立微调。

## 6. 预期 failure modes 与后续迭代方向
- **悬停不下降**：若 progress 信号过弱，可能需要增大 `PROGRESS_WEIGHT` 或加入对 y_position 的额外激励（如垂直接近分项）。v2 可引入垂直 progress 专项。
- **高速撞击**：如果 soft_land_penalty 权重或 threshold 不足，agent 可能在触地时仍有较大速度。可通过提高 `SOFT_LAND_WEIGHT` 或减小 `LANDING_THRESHOLD` 增强末段惩罚。
- **旋转失控**：若 orientation penalty 太轻导致角速度发散，可适量增大 `ANGVEL_WEIGHT`。后续可加入对姿态发动机的适度使用惩罚，防止过度旋转而不点火。
- **垫上未完全稳定（单脚接触）**：目前无接触信号利用，v2 可引入基于双脚接触标志的 soft completion proxy（`joint_condition_proxy`），但需谨慎避免接触奖励黑客。

## 7. 总结
本 v1 奖励函数聚焦于靠近目标垫、低速软着陆、保持直立三个核心信号，完全符合 `reward_role_decomposition` 的主/条件职责划分，避免复杂门控和无效部件，为后续迭代提供清晰的基础。
