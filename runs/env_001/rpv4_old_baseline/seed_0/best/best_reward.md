# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========================== 常数参数 ==========================
    w_progress = 1.0       # 接近目标的主学习信号权重
    w_vx       = 0.05      # 水平速度惩罚权重（平方）
    w_vy       = 0.05      # 垂直速度惩罚权重（平方）
    w_angle    = 0.1       # 姿态角惩罚权重（平方）
    w_ang_vel  = 0.01      # 角速度惩罚权重（平方）
    w_action   = 0.01      # 引擎使用惩罚（离散动作）

    # ========================== 观测解析 ==========================
    # obs / next_obs 均为 shape (8,) 的数组
    x_cur  = obs[0]
    y_cur  = obs[1]
    vx_cur = obs[2]
    vy_cur = obs[3]
    # 当前步的姿态等（未使用，但保留）
    # angle_cur = obs[4]
    # ang_vel_cur = obs[5]

    x_next  = next_obs[0]
    y_next  = next_obs[1]
    vx_next = next_obs[2]
    vy_next = next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    # 接触标志：next_obs[6], next_obs[7]，本版未直接使用

    # ========================== 距离计算 ==========================
    dist_before = (x_cur**2 + y_cur**2) ** 0.5
    dist_after  = (x_next**2 + y_next**2) ** 0.5

    # ========================== 组件 A: 距离缩减奖励（主学习信号） ==========================
    # 鼓励每一步使飞行器向目标着陆垫靠近
    progress_reward = w_progress * (dist_before - dist_after)

    # ========================== 组件 B: 稳定/安全约束（合并多个二次惩罚） ==========================
    # B1. 水平速度惩罚 – 鼓励靠近目标时水平静止
    penalty_vx = -w_vx * (vx_next ** 2)

    # B2. 垂直速度惩罚 – 抑制过大垂直速度（无论上升还是下降过快都危险/浪费燃料）
    penalty_vy = -w_vy * (vy_next ** 2)

    # B3. 姿态角惩罚 – 鼓励保持水平姿态
    penalty_angle = -w_angle * (angle_next ** 2)

    # B4. 角速度惩罚 – 防止剧烈旋转
    penalty_ang_vel = -w_ang_vel * (ang_vel_next ** 2)

    stability_penalty = penalty_vx + penalty_vy + penalty_angle + penalty_ang_vel

    # ========================== 组件 C: 引擎使用效率惩罚 ==========================
    action_penalty = 0.0
    if action != 0:   # 动作 0 表示所有引擎关闭
        action_penalty = -w_action

    # ========================== 总奖励 ==========================
    total_reward = progress_reward + stability_penalty + action_penalty

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'action_penalty': action_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与环境角色

- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **control_type**: discrete（4 个离散动作：0/1/2/3）

## 选用的奖励职责（reward roles）

根据环境卡片中的 `expert_task_profile` 和 `reward_role_decomposition` 的原则，本版聚焦于以下角色：

1. **主学习信号（progress_reward）** – 必须项  
   - **角色**：驱动飞行器飞向目标着陆垫，每步提供梯度。
   - **信号**：`obs[0], obs[1]` 与 `next_obs[0], next_obs[1]` 构成的距离减少。
   - **公式算子**：`improvement_delta`（见 3.2），使用 `dist(obs) - dist(next_obs)`。  
   - **原因**：距离是目标可达性的最直接度量，delta 形式保证 agent 不能停在原地赚取奖励。

2. **稳定/安全约束（stability_penalty）** – 允许项（0~2 个约束，此处合并为一个组件）  
   - **角色**：保证着陆安全，避免高速撞击或翻滚。
   - **所包含的子约束**：
     - 水平速度惩罚 `(-w_vx * vx²)`
     - 垂直速度惩罚 `(-w_vy * vy²)`
     - 姿态角惩罚 `(-w_angle * angle²)`
     - 角速度惩罚 `(-w_ang_vel * ang_vel²)`
   - **公式算子**：`quadratic_penalty`（见 3.4），对所有偏离零点的行为施加平滑、无硬边界的惩罚。
   - **原因**：硬 hinge 依赖准确的阈值（未知环境尺度），quadratic 对小幅偏离也能提供梯度，更适合 v1 阶段的广泛适应性。

3. **引擎使用效率（action_penalty）** – 允许项（0~1 个）  
   - **角色**：满足次要目标——减少燃料消耗。
   - **信号**：离散动作 `action`，非零动作时扣除小量奖励。
   - **公式算子**：`action_efficiency`（见 3.7），固定惩罚每次点火。
   - **原因**：鼓励 agent 用惯性滑翔而非频繁点火，与主目标无冲突（主目标仍可通过接近获得正向奖励）。

## 排除的角色及原因

- **terminal_success_reward / terminal_failure_penalty**：`environment_card.md` 明确声明 `explicit_success_flag_available: false`，`info` 为空，无法安全实现，因此排除。
- **soft_health_gate**：本版采用显式速度/姿态惩罚，而非门控主奖励，以避免 gate 因子选择不当影响探索。若后续出现“先冲后摔”模式，可再加 soft gate。
- **joint_condition_proxy（任务完成近似）**：需要精确的阈值和多种子条件的乘积，v1 阶段参数过多会增加不稳定风险，留到后续迭代（如 v2）加入，以增强最终着陆的引导。
- **potential_based_shaping**：虽然与 improvement_delta 等价，但 delta 形式更简洁，减少过度工程化。

## 训练后应观察的失效模式

- **hover_stall**：agent 学会悬停在目标上方而不下降，因为垂直速度惩罚可能阻止下降。此时需检查垂直速度惩罚权重，或引入下降鼓励项（如奖励向下位移，而非单纯距离减少）。
- **velocity_burst_then_crash**：由于 progress_reward 只奖励靠近，agent 可能全速俯冲，在接近目标时无法减速而坠毁。可通过调整稳定性惩罚权重或未来加入 soft_health_gate 缓解。
- **avoidance_of_engine_use**：动作惩罚过大可能让 agent 完全不用引擎，仅靠初始惯性漂移，无法导航。需监控无引擎步数占比，若过高可调小 `w_action`。
- **excessive_rotation**：角速度或角度惩罚不足时，可能产生旋转刷分？不会，但着陆时姿态不稳定。已包含惩罚。

本设计遵循 `role-based component budget`（主信号 + 稳定约束 + 效率代价，共 3 个组件），所有信号均来自允许使用的 `obs` / `next_obs` 维度，未使用任何未声明的 `info` 字段或 `original_reward`。