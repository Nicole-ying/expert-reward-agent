# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态
    x, y = obs[0], obs[1]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_lc = next_obs[6]
    n_rc = next_obs[7]

    # 距离
    dist_old = (x**2 + y**2)**0.5
    dist_new = (nx**2 + ny**2)**0.5

    # 1. 目标进展：每步距离减少量为正，增加为负
    w_progress = 1.0
    progress = w_progress * (dist_old - dist_new)

    # 2. 软着陆：靠近目标时激活，鼓励低速、水平姿态、双足接触
    landing_threshold = 0.5
    gate = max(0.0, 1.0 - dist_new / landing_threshold)

    w_contact = 0.5
    contact_bonus = w_contact * n_lc * n_rc

    w_speed = 1.0
    speed_penalty = w_speed * (nvx**2 + nvy**2)

    w_angle = 1.0
    angle_penalty = w_angle * (n_angle**2)

    soft_landing = gate * (contact_bonus - speed_penalty - angle_penalty)

    # 3. 燃料效率：对任何使用引擎的动作施以轻微惩罚
    w_fuel = 0.02
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = progress + soft_landing + fuel_cost
    components = {
        "progress": progress,
        "soft_landing": soft_landing,
        "fuel_cost": fuel_cost
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **任务画像**：`navigation_goal_reaching`，动力学子类 `goal_approach_and_soft_contact`，离散控制着陆器。主要目标是尽可能快地从高处到达中央目标板并稳定着陆（位置接近零、速度接近零、双足接触），次要目标节省推力。

- **选用的奖励职责**：
  - `goal_progress`（必选）：使用 `improvement_delta` 算子，每步奖励 `dist_old - dist_new`。驱动智能体持续减小到目标的欧氏距离。
  - `soft_landing`（必选）：在接近目标区（`gate = max(0, 1 - dist/0.5)`）时激活，鼓励双足同时接触（连续乘积给予接触奖励），并惩罚水平和垂直速度平方以及机体倾斜角平方。该组件保证着陆时的稳定性，避免高速撞击或翻倒。
  - `fuel_efficiency`（必选）：使用离散动作代价，对任何非零动作（使用引擎）施加轻微负奖励，促使智能体学习利用惯性滑行和自由落体，减少不必要的推力。

- **排除的职责及原因**：
  - `attitude_stabilization`（条件职责）：虽然重要，但姿态稳定已通过 `soft_landing` 中的角度惩罚隐式处理，无需单独组件，保持了 v1 的简洁。
  - `alive_bonus`、`time_penalty`：缺少存活/时间信号，且与快速到达目标相冲突，因此排除。
  - `contact_early_penalty`：环境没有地形类型标签，过早接触惩罚可能误导学习，已由终端碰撞自然处理。
  - `terminal_success_reward` / `terminal_failure_penalty`：因 `explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，无法从 `info` 获得明确标志，因此不能使用稀疏完成/失败奖励，转而用连续信号构造软引导。

- **未包含但可能后续迭代加入的职责**：
  - 更精细的姿态稳定（如低空强化保持 angle=0）。
  - 动态门控或课程学习：随着训练推进调整 `landing_threshold` 或速度惩罚强度。
  - 边界硬惩罚：若出现大量出界终止，可考虑根据 `x` 位置追加边界惩罚。
  - 动作平滑代价：需要前一步动作历史，当前不可用。

- **训练后应重点观察的 failure modes**：
  - 过度保守（不敢点火，自由坠落导致 crash）：可降低 `w_fuel` 或暂时取消燃料惩罚。
  - 水平漂移出界：加强 `progress` 中水平分量的权重或引入边界惩罚。
  - 目标区上空振荡无法 settle：需增强 `soft_landing` 的速度阻尼或提高接触奖励。
  - 着陆时侧倾翻倒：可后期在 `soft_landing` 中加大角度惩罚或在 `gate` 内单独强化姿态约束。
  - 过度使用主引擎：适当提高 `w_fuel` 或对不同动作施加差异化惩罚。

此版本遵循“从简单到复杂”的原则，用 3 个组件覆盖主学习信号、稳定着陆约束和基础效率要求，为后续迭代留出清晰扩展空间。