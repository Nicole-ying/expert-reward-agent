# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取信号
    x, y = next_obs[0], next_obs[1]
    xv, yv = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_legs = left_contact * right_contact  # 1 如果双脚均接触

    # 当前距离与上一步距离
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (x**2 + y**2) ** 0.5

    # ---- 主进度信号（approach_target）: improvement_delta ----
    progress = dist_prev - dist_next
    w_progress = 0.2

    # ---- 速度阻尼约束（velocity_damping）: 距离门控二次惩罚 ----
    gate = 1.0 / (1.0 + 0.1 * dist_next)          # 越近门控越大
    vel_penalty = xv**2 + yv**2
    w_vel = 0.5

    # ---- 姿态稳定约束（orientation_stabilization）: 二次惩罚 ----
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle = 1.0
    w_angvel = 0.5

    # ---- 软着陆奖励（soft_landing）: 连续条件组合 ----
    prox = 1.0 / (1.0 + 10.0 * dist_next)                     # 靠近目标 ~1
    low_vel = 1.0 / (1.0 + 5.0 * abs(xv) + 5.0 * abs(yv))    # 速度低 ~1
    contact = both_legs                                       # 双脚已接触
    landing = prox * low_vel * contact
    w_land = 5.0

    # 组合
    total_reward = (
        w_progress * progress
        - w_vel * gate * vel_penalty
        - w_angle * angle_penalty
        - w_angvel * angvel_penalty
        + w_land * landing
    )

    components = {
        'progress': w_progress * progress,
        'velocity_damping': -w_vel * gate * vel_penalty,
        'orientation': -w_angle * angle_penalty - w_angvel * angvel_penalty,
        'soft_landing': w_land * landing,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **环境任务画像**  
  `task_family: navigation_goal_reaching`  
  `dynamics_subtype: goal_approach_and_soft_contact`  
  离散控制，严格从观测中推断进展与着陆状态，不使用任何未声明的 info 字段。

- **选用的奖励职责**  
  - `approach_target`（主进度信号）  
  - `velocity_damping`（安全减速约束）  
  - `orientation_stabilization`（姿态稳定约束）  
  - `soft_landing`（任务完成近似信号）  
  共计 4 个组件，符合 v1 的 2–4 组件预算。

- **职责‑信号映射与公式算子**  
  - **approach_target** → `improvement_delta` 算子：  
    `progress = distance(obs) - distance(next_obs)`，鼓励每一步缩短到目标垫的欧氏距离，提供每步稠密梯度。  
  - **velocity_damping** → `quadratic_penalty + soft_health_gate` 变体：  
    使用距离门控 `gate = 1/(1+0.1*dist)` 调节速度惩罚强度：远离目标时惩罚弱（允许高速移动），靠近目标时惩罚增强（强制减速），避免过早减速或快到终点时仍高速。  
  - **orientation_stabilization** → `quadratic_penalty`：  
    直接对机体角度和角速度的平方施加小权重惩罚，防止旋转失控和过大偏转。  
  - **soft_landing** → `joint_condition_proxy` 连续形式：  
    乘积 `proximity_bonus × low_velocity_score × both_legs_contact`，在双脚均接触、位置近、速度低时给出较大奖励；不使用硬阈值，保持梯度用于学习“靠近‑减速‑接触”的协调。

- **排除的职责及原因**  
  - `success_exclusive_bonus` / `terminal_success_reward`：环境无显式成功标志（`explicit_success_flag_available=false`），无法可靠实现。  
  - `terminal_failure_penalty`：`terminated` 信息不可用，且无法从观测直接可靠推断失败类型；同时 v1 优先正向引导而非稀疏惩罚。  
  - `fuel_efficiency_penalty`：属于条件职责，v1 阶段先掌握到达与着陆，效率优化留待后续迭代。  
  - `time_step_penalty`：任务通过着陆后直接终止自然鼓励快速，强制每步惩罚可能与主目标冲突，且缺少量化“浪费时间”的信号。

- **后续迭代可加入的职责**  
  - `fuel_efficiency_penalty`：当 agent 稳定完成任务后，在靠近目标区域时惩罚不必要的引擎点火（基于 `action`），实现隐性燃料消耗最小化。  
  - 更精细的着陆条件（如接触力、下沉量）若环境后续提供相应观测可补充。  
  - 若出现高频摆动或动作振荡，可考虑 `action_smoothness_penalty`（需动作历史，当前未提供）。

- **训练后应重点观察的 failure modes**  
  - 悬停不动：进度信号停滞，但速度惩罚和姿态惩罚弱，需检查 `progress` 分量是否因权重过小或距离门控在远处未提供足够驱动力。  
  - 高速坠向目标垫：`velocity_damping` 的门控在靠近时迅速打开，但主进度信号仍可能诱使高速，需增大 `w_vel` 或调整门控灵敏度。  
  - 持续摇摆，角度振荡：`orientation` 惩罚效果不足，可能需增加 `w_angle` 或引入累积角度误差项。  
  - 提前点燃主引擎导致远离目标：`progress` 会变为负惩罚，但若同时速度惩罚减轻，仍可能出现，后续可增加“远离目标”的额外惩戒。  
  - 仅用姿态引擎旋转不前：速度惩罚过强可能阻碍主引擎使用，需在远离目标时适当放宽速度约束（当前门控已弱化，但权重仍可微调）。
