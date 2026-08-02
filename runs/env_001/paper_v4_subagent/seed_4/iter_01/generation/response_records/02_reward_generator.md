# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x_cur, y_cur = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 超参数
    PROGRESS_WEIGHT = 2.0
    FAIL_PENALTY = -10.0
    LANDING_PROXY_WEIGHT = 2.0
    ANGLE_PENALTY = 0.5
    ANG_VEL_PENALTY = 0.1

    X_BOUNDARY = 1.0
    ANGLE_CRASH = 0.8          # 弧度，约45度
    GROUND_Y_CLOSE = 0.2       # 接近垫面的高度
    VEL_CRASH = 1.5            # 撞击速度阈值
    DIST_LAND = 0.3            # 着陆判定距离范围
    VEL_LAND = 0.5
    ANGLE_LAND = 0.3

    # 1. 进展信号：每步距离的变化量
    dist_cur = (x_cur**2 + y_cur**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)   # 期望 >0

    # 2. 失败惩罚（推断终止原因）
    out_of_bounds = abs(x_next) > X_BOUNDARY
    crash = False
    if (left_contact == 1.0 or right_contact == 1.0):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > ANGLE_CRASH
        high_impact = abs(y_vel_next) > VEL_CRASH
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    failure = out_of_bounds or crash
    failure_penalty = FAIL_PENALTY if failure else 0.0

    # 3. 软着陆近似信号（多条件代理）
    # 距离因子
    dist_to_pad = (x_next**2 + y_next**2) ** 0.5
    dist_factor = max(0.0, 1.0 - dist_to_pad / DIST_LAND)
    # 速度因子
    speed = abs(x_vel_next) + abs(y_vel_next)
    vel_factor = max(0.0, 1.0 - speed / VEL_LAND)
    # 姿态因子
    angle_factor = max(0.0, 1.0 - abs(body_angle_next) / ANGLE_LAND)
    # 接触因子
    contact_factor = 0.5 * (left_contact + right_contact)   # 0, 0.5, 或1

    landing_proxy = (dist_factor + vel_factor + angle_factor + contact_factor) / 4.0
    landing_proxy_reward = LANDING_PROXY_WEIGHT * landing_proxy

    # 4. 姿态/稳定轻惩罚
    stability_penalty = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    total_reward = progress + failure_penalty + landing_proxy_reward + stability_penalty

    components = {
        'progress': progress,
        'failure_penalty': failure_penalty,
        'landing_proxy': landing_proxy_reward,
        'stability_penalty': stability_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与职责选择
- **task_family**: `navigation_goal_reaching`  
- **dynamics_subtype**: `goal_approach_and_soft_contact`  
- **主要目标**: 飞行器尽快飞到中心目标垫并实现软着陆；**次要目标**: 节省燃料（v1暂不作为独立项）。  
- 依据`reward_role_decomposition`，当前v1版本强制执行以下职责：
  - `progress_towards_target`（主学习信号，每步有梯度）
  - `terminal_landing_event`（失败终止惩罚，从观测推断）
  - 条件职责：`orientation_stability`（轻度姿态抑制，作为安全约束）
  - 条件职责：以`joint_condition_proxy`形式实现的**软着陆近似信号**，替代因缺少显式成功标志而无法使用的`terminal_success_reward`。
- **排除的职责**及原因：
  - `terminal_success_reward` → 环境无显式 `success` flag（`explicit_success_flag_available=false`），不能伪造。
  - `terminal_failure_penalty` 本应为强负奖励，但不能使用info字段，故采用从 `next_obs` 推断失败的硬惩罚（仍属于 terminal_event 算子）。
  - `dense_survival_bonus` → 与到达+着陆目标冲突，明确禁用。
  - `fuel_efficiency` → v1阶段暂不加入，避免压制探索；留待后续迭代。

## 2. 职责–信号映射与公式算子
| 职责 | 信号 | 公式算子 | 数学形式与说明 |
|---|---|---|---|
| `progress_towards_target` | `obs[:2]`、`next_obs[:2]`（位置） | `improvement_delta` | `dist_cur - dist_next`，正奖励距离缩短，每步可导。 |
| `terminal_landing_event`（失败） | `next_obs[0]`、`next_obs[4]`、`next_obs[1]`、接触标志（推断失败） | `terminal_event` | 硬惩罚：出界（`|x_next|>1.0`）或撞击（近垫、接触、大倾角/高速）→ `-10.0`。 |
| `soft_contact_behavior` / 任务完成近似 | `next_obs[:4]`、`body_angle`、接触标志 | `joint_condition_proxy` | 四个连续因子（距离、速度、姿态、接触）的算术平均，在有接触且靠近垫时给予高奖励，无接触时基本为0。 |
| `orientation_stability` | `next_obs[4]`（身体倾角）、`next_obs[5]`（角速度） | `quadratic_penalty` | 轻量二次惩罚 `-0.5*angle² - 0.1*ang_vel²`，防止姿态失控。 |

## 3. 为什么没有使用 terminal_success_reward / terminal_failure_penalty（通过info）
- 环境卡片声明 `explicit_success_flag_available=false`，且 info 返回 `{}`，没有任何字段可用；因此无法在终止步直接区分成功/失败。  
- 失败惩罚改为由 `next_obs` 推断：出界或近垫时大倾角/高速接触，给予强负奖励；这种基于观测的`terminal_event`同样提供了明确的失败信号。  
- 成功信号由`landing_proxy`连续化：4个连续因子的乘积/平均鼓励 agent 在所有子条件满足时获得正向反馈，无需一次性大bonus。

## 4. 后续迭代方向
- 引入 `fuel_efficiency`：对离散动作 1/2/3 施加微小成本（主发动机可稍高），并可能使用 curriculum 逐渐增大权重。
- 添加更严格的 `soft_health_gate`：当倾角或坠地风险升高时乘衰减因子到主进展信号，而非简单平行惩罚。
- 如有需要，可加入 `preview_conditioned_reward` 若观测提供前方地面高度信息（当前观测无此维度）。
- 优化超参数阈值（边界、着陆距离、速度门限），根据实际环境尺度调整。

## 5. 预期 failure modes 与观察点
- **早期出界**：如果 `progress` 过强而 `failure_penalty` 不够早，agent 可能冒险高速冲出边界；需观察 `failure_penalty` 的触发频率是否足以抑制此行为。
- **悬停收割 proxy**：`landing_proxy` 在距离略大于阈值但其他条件好时可能给予一定奖励，agent 可能不彻底着陆；需监控 `landing_proxy` 的步均值，若在靠近但未终止时偏高，可收紧阈值或改为乘积形式。
- **姿态惩罚压抑运动**：`stability_penalty` 权重过高可能导致 agent 不敢使用定向引擎，影响姿态调整；需注意该惩罚的量级与主进展奖励的相对大小。
- **失败判断的漏判/误判**：`crash` 条件依赖 `y_next < 0.2` 且接触和倾角/速度，若环境在正常着陆时也短暂满足这些条件，可能错误惩罚；需在少量测试后微调 `ANGLE_CRASH` 与 `VEL_CRASH`。
