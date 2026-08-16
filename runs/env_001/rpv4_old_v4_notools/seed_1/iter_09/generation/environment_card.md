# 匿名环境理解卡片

## 1. 任务目标
控制一台2D飞行器（带主发动机和两侧姿态发动机），从视口顶部中央区域出发，尽快零速平稳降落到中央目标平台，同时尽量少用推进燃料。学习策略需兼顾以下子目标：
- 快速抵近目标平台（x,y 接近零）；
- 下降触地时垂向和水平速度足够低（安全接触）；
- 着陆时保持机身姿态接近水平（角度≈0）；
- 累计推力使用量越小越好。

> 注意：不要将“节省燃料”独立为一个同等优先级目标；它是抵达并安全着陆后的优化指标。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`  
confidence: `high`  
reason: 核心诉求是使飞行器到达目标位置并稳定着陆，属于典型的“到达目标点”导航任务；附属目标（省燃料、快速到达）不改变任务主类型。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64 (从示例看)
- 各维度含义与可用性：
  - obs[0] (x_position)：相对目标平台的水平偏移，`reward_usable: true`
  - obs[1] (y_position)：相对平台高度的垂直偏移，`reward_usable: true`
  - obs[2] (x_velocity)：水平线速度，`reward_usable: true`
  - obs[3] (y_velocity)：垂直线速度，`reward_usable: true`
  - obs[4] (body_angle)：机身俯仰角，`reward_usable: true`
  - obs[5] (angular_velocity)：角速度，`reward_usable: true`
  - obs[6] (left_support_contact)：左支撑腿接触标志（0/1），`reward_usable: true`
  - obs[7] (right_support_contact)：右支撑腿接触标志（0/1），`reward_usable: true`

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 各动作含义：
  - action 0: `no_engine` – 不点火
  - action 1: `left_orientation_engine` – 左侧姿态发动机点火（产生逆时针力矩 + 可能的小横向推力）
  - action 2: `main_engine` – 主发动机点火（产生向上推力 + 可能的旋转影响）
  - action 3: `right_orientation_engine` – 右侧姿态发动机点火（顺时针力矩 + 可能的小横向推力）

## 5. step 与终止条件分析
### 5.1 终止模式
根据提供的终止条件：
- `crash_or_body_contact`：可能为失败（坠毁或危险接触）
- `horizontal_position_outside_viewport`：失败（飞出视野）
- `body_not_awake_or_settled`：身体稳定停住（极可能是成功着陆，即成功终止）

此外，还存在隐式截断（步数上限），但未在任务描述中说明。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: `false` （无 info["success"] 等显式标记）
- explicit_failure_flag_available: `false`
- allowed_info_fields: 空的 `{}`，本环境无可用 info 字段
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（为空）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs`：当前与下一时刻的观测均可使用（`obs`, `next_obs`）
- `action`：当前执行的动作
- `info`：仅内容为空，不可使用任何字段
- `training_progress`：只有明确提示此参数可用时才利用

禁止使用：
- `original_reward`（已屏蔽的官方奖励）
- 未在 `obs`、`action`、`next_obs` 中定义的任何信号
- 任何未明确允许的 info 字段

## 7. 可用于奖励函数的信号
- 位置相关：`x_position (obs[0]/next_obs[0])`、`y_position (obs[1]/next_obs[1])`，可计算目标距离
- 速度相关：`x_velocity (obs[2]/next_obs[2])`、`y_velocity (obs[3]/next_obs[3])`，可用于接触缓速
- 姿态相关：`body_angle (obs[4]/next_obs[4])`、`angular_velocity (obs[5]/next_obs[5])`
- 接触标志：`left_support_contact (obs[6]/next_obs[6])`、`right_support_contact (obs[7]/next_obs[7])`
- 动作：`action (0-3)`，可将推力使用量作为惩罚
- 其他：`next_obs` 可用于方向性变化等

## 8. 不确定或不可用的信号
- 显式成功/失败标记：不存在，不可用
- 与平台精确对齐的额外信息（如平台尺寸）：无
- 剩余燃料或速度上限：无
- 脱离视口的精确边界值：未提供（已知有水平边界，但缺数值，不建议依赖）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 单一刚体（带两条支撑腿）
  actuator_type: 推力器（主发动机 + 两姿态推进器）
  contact_structure: 两个支撑点（左/右）
primary_objectives:
  - 将飞行器中心降至目标平台（x,y ≈0）
  - 以几乎零速、机身水平状态接触并停稳
secondary_objectives:
  - 最小化燃料消耗（减少非零动作）
  - 尽快完成降落（时间效率隐含在燃料/推进选择中）
main_failure_risks:
  - 撞击或高速接触导致坠毁（crash）
  - 水平飞出执行区
  - 悬停/不降落导致截断
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: `goal_proximity`
  purpose: 鼓励飞行器位置向目标点移动，距离越小奖励越高
  why_required: 核心导航目标，缺少则无法引导到达目标
  usable_signals: `[x_position, y_position] from obs/next_obs`
  risks: 若仅用直线距离可能导致不合理的路径（如直接高速俯冲）；需配合速度约束

- role_id: `soft_landing`
  purpose: 在接触或接近平台时，惩罚过大的垂向和水平速度
  why_required: 任务要求安全接触，高速着陆等同失败
  usable_signals: `[x_velocity, y_velocity] from next_obs，以及接触标志 left/right_support_contact`
  risks: 过早惩罚速度可能抑制正常飞行；适合在接近地面时激活

- role_id: `upright_stabilization`
  purpose: 保持 body_angle 接近 0，并抑制角速度
  why_required: 着陆时机身倾斜可能导致倾倒或坠毁
  usable_signals: `[body_angle, angular_velocity] from obs/next_obs`
  risks: 在远离平台时过度要求直立可能妨碍机动

- role_id: `throttle_efficiency`
  purpose: 对使用任何发动机的动作施加惩罚
  why_required: 任务明确要求“as little engine thrust as possible”
  usable_signals: `action`（0 为无推力，其他为推力动作）
  risks: 若惩罚过重可能导致不敢下降；需与其他职责平衡

### 10.2 条件职责 conditional_roles
- role_id: `terminal_settle_boost`
  condition_to_use: 当接触标志显示已着陆（left+right contact=1）且速度接近零时，可给予额外成功信号（如正向奖励），但需谨慎因无显式成功标记
  usable_signals: `[left_support_contact, right_support_contact] & [x_velocity, y_velocity] from next_obs`
  risks: 可能意外奖励非成功停稳状态；强烈建议在训练后期引入或加权极小

### 10.3 慎用/禁用职责 avoid_roles
- role_id: `success_flag_boost`
  reason: 环境中不存在可检测的全局 success/failure 标记，info 为空，无法实现基于官方判断的奖励。
  forbidden_or_missing_signals: `info['success']`, `info['failure']`, `terminated` 原因等

- role_id: `crash_penalty_via_termination`
  reason: 无法在奖励函数中同步获取终止理由，无法直接实现“撞毁扣分”，只能通过速度/接触状态近似。
  forbidden_or_missing_signals: 显式 crash 信号

## 11. role_to_signal_mapping
| role_id                | usable signals                                      | missing signals               | candidate formula operators          | notes |
|------------------------|-----------------------------------------------------|-------------------------------|--------------------------------------|-------|
| `goal_proximity`       | x_position, y_position (obs/next_obs)               | 无                            | dense_state_signal, bounded_distance | 应使用平滑递减（如负距离或高斯） |
| `soft_landing`         | x_velocity, y_velocity, left/right contact (next_obs) | 精确接触力/冲击大小          | velocity_penalty, conditional_scale   | 建议在 y 接近 0 且接触发生前后加大权重 |
| `upright_stabilization`| body_angle, angular_velocity (obs/next_obs)           | 无                            | angle_cost (abs/quadratic), angular_velocity_cost | 可合并为单一姿态误差 |
| `throttle_efficiency`  | action (0-3)                                        | 推力大小（只有开/关）        | action_cost (sparse penalty per nonzero action) | 非零动作均记为消耗 |
| `terminal_settle_boost`| left/right contact + velocities (next_obs)          | 稳健的成功检测               | conditional_bonus (small fixed)       | 极易误判，慎用 |


## 12. 初始训练后应观察的 failure modes
| failure_mode          | evidence_to_check                                              | possible_intervention                            |
|-----------------------|----------------------------------------------------------------|--------------------------------------------------|
| 长时间悬空不下落      | episode长度接近上限，y_position 始终偏高                       | 引导 reward 中加入高度惩罚，或提高主发动机推力使用容忍度 |
| 高速撞击平台          | 终止时 y_velocity 较大，且 crash 终止                          | 强化 soft_landing 角色，调大速度惩罚权重          |
| 姿态失控翻转          | body_angle 或 angular_velocity 持续过大                        | 增强 upright_stabilization，限制角速度           |
| 过量侧移飞出视口      | x_position 超出边界，horizontal_position_outside_viewport      | 增大 horizontal 偏离惩罚，或奖励接近中心线       |
| 极少使用发动机导致未能纠正坠落 | 动作几乎全是 0，y_velocity 持续负增长              | 平衡 throttle_efficiency 和 goal_proximity 权重，避免惩罚过度 |
| 着陆后仍不终止        | 接触标志为 1 但速度非零或微小滑动导致“settled”未触发         | 检查接触后速度，或在接触后降低 soft_landing 权重避免悬停奖励 |