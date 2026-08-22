# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v1 for the lunar‑lander‑like 2D vehicle landing task.
    - Physical components are built only from declared observation signals.
    - No terminal flags or hidden info are used.
    """
    # ---------- goal_proximity: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)   # positive when approaching

    # ---------- attitude_stabilization: penalty for tilt and angular velocity ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_contact: soft proxy for safe two-leg touch-down ----------
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    # Both legs must be in contact to be counted
    contact_score = left_contact * right_contact

    # Low‑velocity factor (both horizontal and vertical speed)
    vx = abs(next_obs[2])
    vy = abs(next_obs[3])
    velocity_factor = 1.0 / (1.0 + 10.0 * (vx + vy))

    # Upright orientation factor (angle near zero)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(angle))

    soft_landing_bonus = 5.0 * contact_score * velocity_factor * angle_factor

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  `navigation_goal_reaching` – `goal_approach_and_soft_contact`，离散推力控制，需要通过渐进式接近和姿态管理实现软着陆。

- **selected reward roles** (按照 `environment_card.md` 的 mandatory 分解)
  - **goal_proximity**（主学习信号）  
    使用 `improvement_delta` 算子：每步奖励距离目标垫中心欧氏距离的减少量 `old_dist - new_dist`。  
    这类信号提供每一时刻的稠密梯度，鼓励持续向目标移动，同时天然惩罚远离行为。
  - **attitude_stabilization**（稳定约束）  
    使用 `quadratic_penalty`：对车身倾角和角速度施加轻量二次惩罚。  
    角色是“方向盘”而非“刹车”，防止大倾角引发碰撞或单腿着陆。
  - **soft_landing_contact**（任务完成近似信号）  
    使用 `joint_condition_proxy`：将双腿接触标志与低速度、小倾角条件组合为乘积形式的软奖励。  
    仅当两块支撑腿同时接触、且速度与姿态均满足“安全”标准时给予正向反馈，避免硬着陆的奖励 hack。

- **role_to_signal_mapping**
  | role | signals used | formula operator |
  |---|---|---|
  | goal_proximity | `obs[0], obs[1], next_obs[0], next_obs[1]` | improvement_delta（距离减少奖励） |
  | attitude_stabilization | `next_obs[4], next_obs[5]` | quadratic_penalty |
  | soft_landing_contact | `next_obs[6], next_obs[7], next_obs[2], next_obs[3], next_obs[4]` | joint_condition_proxy（乘积型软因子） |

- **excluded roles 及原因**
  - `fuel_efficiency`, `time_efficiency`：属于 conditional roles，v1 优先让 agent 学会**安全到达**；能耗与时间优化留待后续迭代，避免过早压制探索或抑制必要引擎使用。
  - `survival_balance`, `sparse_exploration`：按环境卡片明确标为 `avoid_roles`；姿态稳定已内置，无需独立平衡角色，且目标区域已知不需额外探索奖励。
  - `terminal_success_reward` / `terminal_failure_penalty`：  
    `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，info 为空，因此无可靠硬标签可用。本设计完全依靠观测构造的稠密/软信号替代。

- **为什么没有使用 terminal_* 项**  
  环境未提供任何终止状态或成功的显式标志，且 info 字段为空。本 reward 通过 `improvement_delta` 提供持续的接近信号，并通过 `soft_landing_bonus` 在满足条件时给予接近“成功语义”的正向激励，以此弥补稀疏终端信号的缺失。

- **后续迭代预留**
  - `fuel_efficiency`：可在 v2 中引入小幅动作惩罚（`action_count_penalty`），约束无效点火。
  - `time_efficiency`：可加入极小的常量步惩罚加速下降，但需配合高度下降激励避免滞空。
  - 可引入 `soft_health_gate` 若出现“高速撞垫”仍得正奖励的情况，以高度/速度门控主信号。
  - 如有需要可增加基于角度变化率的姿态平滑项。

- **训练后应重点观察的 failure modes**
  - **悬停不动**：若 agent 高度长期不变且 progress 奖励接近零，可能是因为惩罚项太强而 progress 奖励太弱；可增大 `proximity_progress` 权重或引入小量下降激励。
  - **高速撞击垫面**：如果 terminate 时 `|vy|` 很大，`soft_landing_bonus` 无法及时激活；可考虑加入基于速度的惩罚或在高速接触时额外惩罚。
  - **单腿着陆/侧翻**：乘积式 `contact_score` 鼓励双腿接触，但若姿态惩罚过弱仍可能出现；可调整 `attitude_penalty` 或角度因子强度。
  - **引擎滥用导致姿态振荡**：`attitude_penalty` 直接抑制角速度，可适当增大其权重以避免频繁的点火/回正循环。
  - **飞出水平边界**：当前 reward 仅用距离惩罚，未专门处理边界犯规；后续可在边界附近加入强惩罚或截断信号。
