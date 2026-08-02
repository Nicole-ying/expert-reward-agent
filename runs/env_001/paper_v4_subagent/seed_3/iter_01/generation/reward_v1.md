# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observation indices (assumed 2D lander, similar to LunarLander‑v2)
    # 0: x position, 1: y position, 2: vx, 3: vy, 4: angle, 5: angular velocity
    # 6: left leg contact, 7: right leg contact (0.0 or 1.0)
    # Target landing pad is assumed at (0.0, 0.0)

    # Extract current and next states
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle, n_ang_vel = next_obs[4], next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # ---------- 1. Main progress: distance to target decreasing ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new              # + when approaching target

    w_progress = 10.0

    # ---------- 2. Stability constraint: attitude & angular velocity ----------
    # Quadratic penalty on tilt and rotation
    w_angle = 0.5
    w_angvel = 0.1
    stability = -w_angle * (n_angle ** 2) - w_angvel * (n_ang_vel ** 2)

    # ---------- 3. Lateral drift constraint: horizontal speed ----------
    w_lat = 0.2
    lateral_drift = -w_lat * (nvx ** 2)

    # ---------- 4. Soft landing bonus (joint‑condition proxy) ----------
    landing_bonus = 0.0
    # Conditions: both legs on ground, nearly upright, gentle vertical speed
    if n_left > 0.5 and n_right > 0.5 and abs(n_angle) < 0.1 and abs(nvy) < 0.2:
        landing_bonus = 10.0

    # Combine
    total_reward = (w_progress * progress
                    + stability
                    + lateral_drift
                    + landing_bonus)

    components = {
        "progress": w_progress * progress,
        "stability_penalty": stability,
        "lateral_drift_penalty": lateral_drift,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像（基于环境卡片推断）
- **task_family**：2D 飞行器着陆（2D Lander）  
- **dynamics_subtype**：连续控制，受重力和推力，从视口顶部启动，需平稳降落在中央平台。  
- **目标**：尽快、平稳地到达并稳定停留在着陆点（目标平台中心）。  
- **观测空间假设**：  
  由于环境卡片未提供精确的观测维度，本设计参照经典 2D 着陆器（如 LunarLander‑v2）做出以下合理假设：
  - 位置 `(x, y)`，速度 `(vx, vy)`，角度 `(angle)`，角速度 `(ang_vel)`
  - 左/右着陆腿接触标志 `(left_contact, right_contact)`（连续值 0/1）
  - 目标平台位于坐标原点 `(0,0)`
  - 若实际观测结构不同，需相应调整索引。

## 2. 奖励职责拆解（selected reward roles）
根据“尽快、平稳降落”这一目标，v1 版本选取以下职责：

| 职责 | 角色 | 设计动机 |
|------|------|----------|
| **主学习信号** | 距离减少（progress） | 提供每步梯度，直接将 agent 拉向目标。距离变化量是稠密且与任务目标直接相关的信号。 |
| **稳定/安全约束** | 姿态 & 角速度惩罚（stability） | 防止过度倾斜或旋转，保证平缓飞行。 |
| **侧向漂移约束** | 水平速度惩罚（lateral drift） | 抑制水平滑移，使 agent 学会垂直着陆，不冲出平台。 |
| **任务完成近似信号** | 软着陆奖励（landing bonus） | 利用腿接触 + 低垂直速度 + 小角度构造安全的着陆条件，补偿因缺少显式 success flag 的信号缺失。 |

## 3. 职责‑信号映射与公式算子选择
- **progress**  
  信号：`(x, y)` → 距离 `dist`  
  算子：**improvement_delta**（`dist_old − dist_new`），鼓励距离随时间单调减小。  
  *理由*：距离变化比距离绝对值更能驱动持续靠近目标，避免停滞。

- **stability**  
  信号：`angle` (obs[4])、`ang_vel` (obs[5])  
  算子：**quadratic_penalty** (`−w * value²`)，对所有偏离零的值给予连续惩罚。  
  *理由*：没有明确的硬边界，二次惩罚足以抑制大幅摆动。

- **lateral_drift**  
  信号：`vx` (obs[2])  
  算子：**quadratic_penalty**，惩罚水平速度的平方。  
  *理由*：水平移动非必要，留在原地更符合“稳定停留在着陆点”的要求。

- **landing_bonus**  
  信号：`left_contact`, `right_contact` (obs[6],[7])，`n_angle`，`nvy`  
  算子：**joint_condition_proxy**（多条件逻辑与 + 连续阈值），满足：双腿触地、`|angle|<0.1`、`|vy|<0.2` 时给予固定正奖励。  
  *理由*：没有显式 success flag，通过多条连续条件的组合构造一个可靠的“软成功”信号，引导最终着陆动作。

## 4. 排除的职责与原因
- **terminal_success_reward / terminal_failure_penalty**  
  环境卡片未声明 `explicit_success_flag_available` 或 `explicit_failure_flag_available`，info 内容未知，故不依赖任何 `info` 字段。
- **效率/动作代价**（如推力能量惩罚）  
  v1 阶段优先学习“朝目标靠近并安全着陆”的基本行为，能耗优化留待后续迭代。
- **soft_health_gate / preview_conditioned_reward**  
  当前失败模式主要源于高速坠毁或翻转，通过姿态与速度惩罚已可缓解，暂时无需引入更复杂的主奖励门控。
- **动态课程/门控**  
  训练初期缺乏历史数据，不适合引入依赖训练进度的权重调整。

## 5. 后续迭代计划（不在 v1 中实现）
- 加入**推力效率/能耗约束**（agent 学会着陆后可能浪费燃料来回摆荡）。
- 若观测中包含高度信息，可引入**接近地面时的引导项**（如 soft_health_gate）。
- 观察`landing_bonus`被滥用的情况，必要时代换为更加平滑的 potential‑based shaping。

## 6. 训练后应重点观察的失败模式
- **悬停而不着陆**：agent 学会在目标上方保持高位，距离变化趋零，但没收起落架——说明 `landing_bonus` 权重不足或条件过严。
- **高速俯冲坠毁**：`progress` 权重过大导致 agent 以极快速度冲向目标，尽管有姿态惩罚仍可能失控。需监控垂直速度分布。
- **利用`landing_bonus`刷分**：agent 可能反复小幅度弹跳，多次触发 bonus。可通过将 bonus 固定为仅在首次接触时触发，或改为一次性奖励（需引入状态记录，v2 再考虑）。
- **水平漂移振荡**：`lateral_drift` 惩罚可能造成过度抑制，agent 不敢调整水平位置，反而错过着陆。需检查 `vx` 的分布和实际着陆点。
- **不必要的大幅动作**：缺少动作代价时，agent 可能使用剧烈推力，虽能完成任务但效率低下——留给后续效率项解决。