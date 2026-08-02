# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆分
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft_contact = next_obs[6]
    nright_contact = next_obs[7]

    # 超参数
    w_progress = 5.0
    w_landing = 2.0
    w_land_vel = 10.0
    w_angle = 0.5
    w_angvel = 0.5
    engine_cost = 0.02

    # 距离计算
    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5

    # 1. 主学习信号：距离改进（potential‑based shaping）
    progress = w_progress * (dist - ndist)

    # 2. 着陆质量软信号（仅在双腿接触时激活）
    contact = nleft_contact * nright_contact  # 0 或 1
    x_thresh = 0.1
    vx_thresh = 0.2
    vy_thresh = 0.2
    angle_thresh = 0.1

    fx = max(0.0, 1.0 - abs(nx) / x_thresh)
    fvx = max(0.0, 1.0 - abs(nvx) / vx_thresh)
    fvy = max(0.0, 1.0 - abs(nvy) / vy_thresh)
    fangle = max(0.0, 1.0 - abs(nangle) / angle_thresh)
    fcontact = float(contact)

    if fcontact > 0.5 and fx > 0 and fvx > 0 and fvy > 0 and fangle > 0:
        landing_quality = (fcontact * fx * fvx * fvy * fangle) ** (1.0 / 5.0)
    else:
        landing_quality = 0.0
    landing_reward = w_landing * landing_quality

    # 3. 着陆速度惩罚（仅在接触时）
    if fcontact > 0.5:
        vel_pen = -w_land_vel * (nvx**2 + nvy**2)
    else:
        vel_pen = 0.0

    # 4. 姿态稳定惩罚（全程）
    att_penalty = -w_angle * (nangle**2) - w_angvel * (nangvel**2)

    # 5. 引擎使用惩罚（节省燃料）
    eng_pen = -engine_cost if action != 0 else 0.0

    total_reward = progress + landing_reward + vel_pen + att_penalty + eng_pen
    components = {
        "progress": progress,
        "landing_quality": landing_reward,
        "landing_velocity_penalty": vel_pen,
        "attitude_penalty": att_penalty,
        "engine_cost": eng_pen
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与动力学子类型
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **控制类型**: 离散 4 动作（无、左姿态、主引擎、右姿态）
- **核心目标**: 飞行器安全、稳定地降落在平台中心，双腿接触，姿态水平，速度接近零。
- **次要目标**: 节省引擎使用。

## 2. 已选择的奖励角色（selected reward roles）
基于 `reward_role_decomposition` 中的 **mandatory roles** 和一项 **conditional role**，共 5 个组件：

| 角色 | 对应组件 | 使用的信号 |
|------|----------|------------|
| `goal_distance_shaping` | `progress` – 距离改进差 | `x`, `y`, `nx`, `ny` |
| `safe_landing_contact` | `landing_quality` – 连续着陆质量（乘积几何平均） | 双腿接触 + 水平位置、速度、姿态角（`nx`, `ny`, `nvx`, `nvy`, `nangle`, `nleft_contact`, `nright_contact`） |
| `velocity_penalty_at_landing` | `landing_velocity_penalty` – 只在接触时惩罚速度平方和 | 接触标志 + 速度 |
| `attitude_stabilization` | `attitude_penalty` – 全程姿态/角速度惩罚 | 角度、角速度 |
| `engine_usage_penalty` | `engine_cost` – 每非零动作固定小惩罚 | `action` |

### 公式算子选择
- **progress**: 使用 `improvement_delta`（即潜在成形 `old_distance - new_distance`），提供每步向目标靠近的梯度。
- **landing_quality**: 使用 `joint_condition_proxy`，将多个硬着陆条件（双腿接触、近中心、低速、水平姿态）转化为连续几何平均因子，保证即使单个条件趋零，整体奖励仍连续。
- **landing_velocity_penalty**: `quadratic_penalty` 且仅在接触时激活，避免抑制飞行中的必要机动。
- **attitude_penalty**: 全程 `quadratic_penalty`，轻量约束姿态与角速度。
- **engine_cost**: 固定步级惩罚，对应 `action_efficiency` 思路。

## 3. 排除的角色及原因
- **`terminal_success_reward`** / **`terminal_failure_penalty`**：`explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，且无法访问 `done` 或 `info` 中的终止原因。我们已通过 per‑step 着陆质量信号替代了终端成功奖励。
- **`approach_path_smoothness`**：需要前一步动作，没有记忆能力，v1 不引入。
- **`survival_bonus`**、**`exploration_bonus`**、**`explicit_timer_penalty`**：与任务性质冲突或信号不可用，已按环境卡片排除。

## 4. 为何不使用 terminal_success_reward / terminal_failure_penalty
环境完全隐藏了终止原因（`info` 为空），且 `done` 未传入奖励函数。即便能在 `next_obs` 中推断着陆成功状态，我们也无法判断该步是否为 episode 的最后一帧。因此改用 per‑step 着陆质量软信号 `landing_quality`，它在双腿接触且满足关键平缓条件时持续提供正向激励，引导 agent 达到并保持安全着陆状态（episode 随后自然终止）。

## 5. 留到后续迭代的职责
- **动态引擎惩罚权重**：可根据距平台的距离动态调整，更精细地权衡燃料与机动。
- **自适应姿态约束**：在着陆阶段加强角度限制，飞行阶段放松。
- **课程式训练进度接入**：当前未使用 `training_progress`，未来若需逐步收紧着陆阈值或增加速度惩罚，可引入。
- **软终止失败信号**：若能从观测推断 crash/出界，可在 v2 加入 `terminal_event` 硬覆盖惩罚，进一步提升可靠性。

## 6. 训练后应观察的失效模式
- **悬停空中不肯下降**：检查 `progress` 和 `y` 位置变化，可能需要增大垂直接近的梯度。
- **高速冲击平台后终止**：虽然速度惩罚在接触时生效，但若 agent 仍以高速触地，需增大 `w_land_vel` 或调整阈值。
- **单腿着陆、姿态倾斜**：检查 `attitude_penalty` 和 `landing_quality` 是否足够强，必要时增加姿态权重或使用更灵敏的角度因子。
- **过度使用主引擎徘徊**：查看 `engine_cost` 是否过低，可在接近平台区域增加局部惩罚。
- **横向漂出视口**：监测水平位置是否持续增大，可能需要加强横向距离的惩罚或调整进度信号的几何结构。
