# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs and next_obs are both length 8 vectors
    # indices: 0:x_position, 1:y_position, 2:x_velocity, 3:y_velocity,
    #          4:body_angle, 5:angular_velocity, 6:left_support_contact, 7:right_support_contact
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]

    next_x, next_y = next_obs[0], next_obs[1]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # Current and next distances to target pad (0,0)
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # ------------------ 1. Progress reward (main learning signal) ------------------
    w_progress = 2.0
    progress = dist - next_dist          # positive when moving closer
    progress_reward = w_progress * progress

    # ------------------ 2. Stability penalty (gated constraint) ------------------
    gate = 1.0 / (1.0 + dist)            # weak when far, strong when close
    w_vel = 0.2
    w_ang = 0.2
    stability_error = w_vel * (vx ** 2 + vy ** 2) + w_ang * (angle ** 2 + angvel ** 2)
    stability_penalty = -gate * stability_error

    # ------------------ 3. Fuel penalty (efficiency) ------------------
    w_fuel = 0.02
    fuel_penalty = -w_fuel if action != 0 else 0.0

    # ------------------ 4. Safe contact bonus (soft completion proxy) ------------------
    w_contact = 0.5
    # continuous proximity factor: 1 when distance -> 0, ~0 when distance large
    prox_factor = 1.0 / (1.0 + 10.0 * next_dist)
    # only active when both legs touch
    safe_contact_bonus = w_contact * prox_factor * next_left * next_right

    # ---------------------------------------------------------------------------------
    total_reward = progress_reward + stability_penalty + fuel_penalty + safe_contact_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "fuel_penalty": fuel_penalty,
        "safe_contact_bonus": safe_contact_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像
- **task_family**: `navigation_goal_reaching`  
- **dynamics_subtype**: `goal_approach_and_soft_contact`  
- **控制类型**: 离散动作（4 个引擎选项）  
- **核心目标**: 安全着陆到中央目标垫并稳定停驻；次要节省燃料。

## 2. 选用的奖励职责与信号映射
根据 `environment_card.md` 的 `reward_role_decomposition`，在可用信号范围内选择了四个职责：

| 职责角色 | 对应的公式算子 | 使用的信号 |
|----------|----------------|-------------|
| `goal_proximity_reward` (主学习信号) | `improvement_delta` | obs[0,1], next_obs[0,1] → 距离减少 |
| `stability_penalty` (稳定约束) | `quadratic_penalty` + 位置门控 | obs[2], obs[3], obs[4], obs[5] |
| `fuel_thrust_penalty` (效率惩罚) | 离散动作惩罚 | `action` (0‑3) |
| `safe_contact_bonus` (条件完成近似) | `joint_condition_proxy`（连续接近因子 × 接触标志） | next_obs[6,7] 和 next_obs[0,1] 计算的距离 |

## 3. 各组件详细说明
### 3.1 主学习信号：`progress_reward`
- **形式**: `w * (current_distance - next_distance)`  
- **理由**: 每一步距离减少都得到正反馈，直接引导向目标移动；相比于 `-w*dist` 更能强调进步，且不会在远处给出过强的负信号。  
- **风险控制**: 到达目标后增量接近零，不再有大量奖励，需由稳定惩罚和接触奖励鼓励停驻。

### 3.2 稳定约束：`stability_penalty`
- **形式**: `-gate * (w_vel*(vx^2+vy^2) + w_ang*(angle^2+angvel^2))`，其中 `gate = 1/(1+dist)`。  
- **理由**: 使用位置门控使远离目标时惩罚极轻，保留探索自由；接近目标时逐渐加强，强制减速和姿态收敛。  
- **算子**: `quadratic_penalty` + 门控（来自 `bounded_signal` 思路）。避免了全局强惩罚导致 agent 不敢高速靠近。

### 3.3 燃料惩罚：`fuel_penalty`
- **形式**: 若 `action != 0` 则负奖励 `-0.02`。  
- **理由**: 任务明确要求节省燃料，且离散动作使非零动作直接对应推力消耗。权重极小，不会压制主任务（agent 在需要点火时不致因惩罚而放弃）。  
- **v1 策略**: 仅保留恒定小惩罚，后续迭代可引入动作平滑或更高权重的效率项。

### 3.4 安全接触奖励：`safe_contact_bonus`
- **形式**: `w_contact * (1/(1+10*next_dist)) * left_contact * right_contact`。  
- **理由**: 当且仅当两腿同时触地且距离目标很近时给予额外奖励，近似“成功着陆”信号。连续距离因子提供梯度，避免纯离散条件。  
- **风险处理**: 乘子 `next_dist` 保证远离垫子时即使接触，奖励也微乎其微；防止在远处撞地时恶意刷分。

## 4. 排除的职责及原因
- **`termination_success_reward`**: 环境中没有显式成功标志（`info` 为空），无法区分成功终止与平凡静止。  
- **`terminal_failure_penalty`**: 同样缺少失败标志，不能在 `compute_reward` 内可靠判断失败。  
- **动态课程/门控奖励 (`curriculum_weighting`)** : `training_progress` 已被禁止使用，且 v1 不需要多阶段权重。  
- **动作平滑惩罚**：没有提供历史动作序列，无法实现。  
- **强推力优化**：未使用剩余燃料或推力大小信号，仅保留简单的动作计数惩罚。

## 5. 为什么没有使用 terminal success/failure 奖励
- `explicit_success_flag_available = false`, `explicit_failure_flag_available = false`。  
- `info` 固定为空，无法获得 `termination_reason`。  
- 强行假设成功或失败可能导致错误奖励分布，故完全避免。

## 6. 留到后续迭代的职责
- **更精细的效率代价**：例如分动作惩罚、点火次数平滑约束。  
- **接触安全性的动态验证**：结合速度阈值，防止高速撞击时的“接触”奖励。  
- **课程式门控**：前期允许更鲁莽的探索，后期收紧稳定阈值。  
- **生存/时间奖励**：无截断环境，暂不显式鼓励快速完成；未来可引入与时间步数相关的轻量信号。

## 7. 训练后应观察的失败模式
1. **高速坠毁**：`stability_penalty` 的 gate 可能仍不足以在接近末期强制减速，agent 可能以高速撞击垫子。  
2. **原地怠速/不敢点火**：燃料惩罚过大或稳定惩罚过早压制可能导致 agent 选择不点火并停留在远处。  
3. **伪着陆**：agent 可能在远离目标处用支撑腿触地触发极小的 `safe_contact_bonus`，并依靠该奖励原地踏步（因为位置进步奖励此时由稳定惩罚抵消）。需检查 reward 分解值。  
4. **姿态振荡**：离散动作下，纠正角度的代价可能引发左右引擎反复切换，产生高频 angular velocity，导致 `stability_penalty` 上升。  
5. **单腿着陆**：agent 可能满足于单腿接触，没有完全稳定就结束 episode（由环境终止逻辑判断），此时 `safe_contact_bonus` 无法引导双腿同时触地。  
6. **飞出视口**：水平位置超出界限直接终止，但奖励中没有水平位置惩罚；v1 依靠距离奖励中的 x 分量间接抑制，可能需要后续添加边界软惩罚。
