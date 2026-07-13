# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox   = 1.0   # distance reduction weight
    w_vel    = 0.3   # landing velocity penalty weight
    w_angle  = 0.2   # body angle penalty weight
    w_angvel = 0.1   # angular velocity penalty weight
    w_bonus  = 0.5   # stable‑stand bonus weight

    k_gate_y = 20.0  # steepness of height gate
    k_fx     = 5.0   # factor scaling for horizontal distance
    k_fy     = 5.0   # factor scaling for vertical height
    k_fspeed = 2.0   # factor scaling for speed
    k_fangle = 2.0   # factor scaling for body angle

    # ---------- 1. progress: distance reduction ----------
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress  = prev_dist - next_dist
    progress_reward = w_prox * progress

    # ---------- 2. soft landing velocity penalty ----------
    # gate = 1 when y is near zero (height close to pad), decays when high
    y_clipped = max(0.0, next_obs[1])   # y positive above pad
    gate_vel  = 1.0 / (1.0 + k_gate_y * y_clipped)
    speed     = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    landing_vel_penalty = -w_vel * speed * gate_vel

    # ---------- 3. upright stability penalty ----------
    upright_penalty = -w_angle * abs(next_obs[4]) - w_angvel * abs(next_obs[5])

    # ---------- 4. stable‑stand bonus (joint condition proxy) ----------
    # each factor ∈ [0,1]; geometric mean avoids collapse
    factor_x       = 1.0 / (1.0 + k_fx * abs(next_obs[0]))
    factor_y       = 1.0 / (1.0 + k_fy * max(0.0, next_obs[1]))
    factor_speed   = 1.0 / (1.0 + k_fspeed * speed)
    factor_angle   = 1.0 / (1.0 + k_fangle * abs(next_obs[4]))
    factor_contact = next_obs[6] * next_obs[7]  # 1 only when both legs touch

    prod = factor_x * factor_y * factor_speed * factor_angle * factor_contact
    stable_bonus = w_bonus * (prod ** 0.2) if prod > 0.0 else 0.0

    # ---------- total reward ----------
    total_reward = progress_reward + landing_vel_penalty + upright_penalty + stable_bonus

    components = {
        'progress': progress_reward,
        'landing_vel_penalty': landing_vel_penalty,
        'upright_penalty': upright_penalty,
        'stable_bonus': stable_bonus
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 选定的任务画像与动力学子类
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- 控制方式为离散动作，机体是带推力和接触传感器的2D刚体。

## 奖励职责选择
依据 `reward_role_decomposition`，v1 版本包含以下角色：
1. **主学习信号** ─ `proximity_to_target`：使用 `improvement_delta` 算子，奖励每一步到目标垫板中心欧氏距离的减小量。
2. **软着陆速度惩罚** ─ `soft_landing_velocity_penalty`：使用 `conditional_quadratic_penalty` 思路，通过高度门控乘上合速度大小进行惩罚，仅在接近地面（y 小）时生效并强制减速。
3. **直立稳定性惩罚** ─ `upright_stability_penalty`：使用 `absolute_value` 对机身角度和角速度施加全局轻量惩罚，避免倾覆。
4. **稳定停泊奖励** ─ `stable_stand_bonus`（条件职责）：使用 `joint_condition_proxy`，由水平偏差、高度、速度、姿态和双足接触五个连续因子通过几何平均构建软完成代理，鼓励最终牢牢停在垫板上。

## 角色‑信号映射与公式算子
| role | 信号 | 算子 |
|---|---|---|
| proximity_to_target | `obs[0:2]`, `next_obs[0:2]` 距离 | `improvement_delta`: `prev_dist - next_dist` |
| soft_landing_velocity_penalty | `next_obs[1]`（高度）, `next_obs[2:4]`（速度） | `multiplicative_activation`: `-w_vel * speed / (1 + k*y)` |
| upright_stability_penalty | `next_obs[4]`（角度）, `next_obs[5]`（角速度） | `absolute_value` 组合 |
| stable_stand_bonus | 位置、高度、速度、角度、接触标志 | 几何平均的 `joint_condition_proxy`：`(prod)^0.2` |

## 排除的角色及原因
- **explicit_success_shaping**：环境 `info` 为空字典，无显式成功标志，无法安全使用 terminal_success_reward 或 terminal_failure_penalty。
- **remaining_time_reward**：未提供时间步计数器，无法构造时间相关奖励。
- **fuel_efficiency_penalty**：动作代价型职责留待后续迭代，v1 先让智能体学会到达和软着陆。
- **curriculum_weighting**：v1 尚无训练进度参考，且任务并非早期‑后期存在根本冲突。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
因为 `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，且 `info` 固定为空。强行用硬编码条件替代官方成功标志会引入假阳性，与真实物理终止不一致，违反信号可用性原则。

## 留在后续迭代的职责
- **燃料效率惩罚**（`fuel_efficiency_penalty`）：当接近目标且速度足够低时，惩罚非必要引擎点火。
- **动作平滑性代价**：由于没有历史动作，无法在 v1 实现。
- **更复杂的门控与动态课程**。

## 训练后应重点观察的失败模式
- 距离减小的驱动力可能让智能体高速冲向垫板，需观察 **crash 终止率** 是否因速度惩罚不足而偏高。
- 高度门控若设置过强，可能使智能体悬停在低空而迟迟不落地，需检查 **episode 长度** 与 **progress 停滞**。
- 直立惩罚权重过大可能阻碍姿态调整，表现为 **单腿着地后反复弹跳**。
- 稳定奖励的几何平均要求所有因子同时较高，若智能体长时间无法获得 bonus，需检查其是否因接触因子为 0 而无法得到引导，后期可考虑改用算术平均或分开奖励双触。