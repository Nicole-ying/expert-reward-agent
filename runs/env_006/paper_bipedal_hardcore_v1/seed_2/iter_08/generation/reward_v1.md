# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract useful signals ----------
    horizontal_speed = obs[2]          # forward direction
    hull_angle = obs[0]                # tilt (rad)
    hull_angular_velocity = obs[1]     # tilt speed (rad/s)

    # ---------- health gate: close to 1 when upright, decays when tilting ----------
    #   gate = 1 / (1 + 10 * angle^2 + 0.1 * angvel^2)
    #   avoids over-punishing early exploration, but heavily cuts progress reward
    #   when tilt becomes dangerous.
    denom = 1.0 + 10.0 * hull_angle * hull_angle + 0.1 * hull_angular_velocity * hull_angular_velocity
    health_gate = 1.0 / denom

    # ---------- forward progress (only positive direction) ----------
    #   only reward moving forward; ignore backward motion (max to avoid penalizing it)
    fwd_speed = max(0.0, horizontal_speed)
    progress_component = 1.0 * fwd_speed * health_gate   # w_speed = 1.0

    # ---------- moderate action regularisation ----------
    #   small penalty on large joint torques – just enough to avoid extreme signals
    action_sum_sq = action[0]*action[0] + action[1]*action[1] + action[2]*action[2] + action[3]*action[3]
    action_penalty = -0.01 * action_sum_sq

    # ---------- total reward ----------
    total_reward = progress_component + action_penalty

    components = {
        "progress": progress_component,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与职责选择
- **task_family**: `locomotion_continuous_control`  
- **dynamics_subtype**: `planar_bipedal_gait`  
- **selected reward roles**（按 `reward_role_decomposition`）:
  - **forward_progress**：核心驱动力，让 agent 持续向前。  
  - **survival_balance**：防止摔倒，保证运动安全。  
  这两个是 mandatory roles，缺失任何一项都会导致策略失败。

## 2. 职责–信号映射
| role | signal(s) | 可用性 |
|---|---|---|
| forward_progress | `horizontal_speed` (obs[2]) | 每步可用，连续正梯度 |
| survival_balance | `hull_angle` (obs[0])、`hull_angular_velocity` (obs[1]) | 每步可用，直接反映倾倒风险 |

## 3. formula operator 选择
- **主进度信号**：`dense_state_signal` 的线性正向奖励 `w * signal`，但通过 **soft_health_gate** 动态调制，防止“高速前进＋高倾斜”的组合。  
  - gate 形式：倒数门 `1 / (1 + k1*angle² + k2*angvel²)`，倾斜越严重，gate 衰减越猛烈，避免主奖励在危险状态下仍正向驱动。
- **健康约束**：未使用独立惩罚（hinge/quadratic），而是用 gate **乘入** 主奖励。原因：
  - 之前的尝试（独立 hinge_penalty、balance_penalty 作为加法项）均无法突破 -52 分；agent 可能在倾斜时仍获得较大的前进奖励，导致惩罚被主奖励淹没。  
  - 将健康信号做成乘法 gate，能在身体恶化时直接削减前进奖励，语义更明确：  
    “状态不好时，不允许再利用高速度刷分”。
- **动作效率**：`quadratic_penalty` `-0.01 * Σa_i²`，极小权重，仅抑制极端力矩，不影响探索。

## 4. excluded roles 及原因
- **terminal_success_reward / terminal_failure_penalty**：环境中 `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，`info` 为空。虽然可推断摔倒或到达终点，但推断不可靠，容易引入噪声，v1 阶段不采用。
- **terrain_gate / preview**：激光雷达 (obs[14:23]) 虽可用，但映射到奖励的函数难以设计，且 v1 应优先解决“前进‑平衡”的核心矛盾。
- **关节力矩能耗**：动作惩罚已覆盖。

## 5. 对过往失败尝试的反思
历次尝试（forward_reward + 独立平衡/角度惩罚）最高分仅 -52.19，说明 **独立加法惩罚无法说服 agent 在收益与风险间正确权衡**。本版改用 **soft_health_gate**，将平衡约束整合进进度奖励本身，使 agent 面临直接的机会成本：越倾斜 → 越难拿分。这是对过往结构的根本性改变，有望打破停滞。

## 6. 留待后续迭代的职责
- 精确的摔倒终端惩罚（如果后续可以可靠推断）。  
- 地形预判 gate（利用 laser 数据）。  
- 动态课程或 training_progress 调整的权重。

## 7. 训练后应观察的 failure modes
- agent **原地站立不前进**（gate≈1，速度≈0）——说明动作惩罚可能相对过强，或需要略微提高 `w_speed`。  
- agent **在平坦段高速奔跑但遇障碍仍不减速，导致摔倒**——说明 gate 的衰减区间还需调整（增加 `k1` 或引入角速度项）。  
- 出现**高频小幅度摇晃**但未摔倒——动作惩罚应仍能轻微抑制，无需额外组件。